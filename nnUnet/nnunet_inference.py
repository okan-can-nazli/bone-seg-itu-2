import os
import json
import subprocess
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from Unet.dataset import load_merged_mask, build_file_lists

#########################################################################################################
#! Constants

DATASET_ID   = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_BoneSeg"

#! Directories

# LOCAL DIRS
# DATA_DIR        = "Data"
# BASE_DIR        = "nnunet_working"

# KAGGLE DIRS
DATA_DIR        = "/kaggle/input/datasets/foxcancoy/bone-seg-2/Data"
BASE_DIR        = "/kaggle/working"

RAW_DIR         = os.path.join(BASE_DIR, "nnunet_raw")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "nnunet_predictions")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

NNUNET_RAW          = RAW_DIR
NNUNET_PREPROCESSED = os.path.join(BASE_DIR, "nnunet_preprocessed")
NNUNET_RESULTS      = os.path.join(BASE_DIR, "nnunet_results")

CONFIG  = "2d"
TRAINER = "nnUNetTrainer_250epochs"

TEST_SPLIT = 0.2  # fraction of data to hold out as test set
#########################################################################################################


# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────

def install_nnunet():
    """Install nnU-Net v2 if not already available."""
    try:
        import nnunetv2
        print("nnU-Net already installed.")
    except ImportError:
        print("Installing nnU-Net v2...")
        subprocess.run(["pip", "install", "nnunetv2"], check=True)
        print("nnU-Net installed.")


def set_env():
    """Set required nnU-Net environment variables."""
    os.environ["nnUNet_raw"]          = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"]      = NNUNET_RESULTS

    os.makedirs(NNUNET_RAW,          exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)
    os.makedirs(NNUNET_RESULTS,      exist_ok=True)

    print("Environment variables set.")
    print(f"  nnUNet_raw          = {NNUNET_RAW}")
    print(f"  nnUNet_preprocessed = {NNUNET_PREPROCESSED}")
    print(f"  nnUNet_results      = {NNUNET_RESULTS}")


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_nnunet_dataset():
    """
    Convert PNG X-rays + JSON masks to nnU-Net NIfTI format.
    Splits data into train and test sets.
    Returns list of (case_id, gt_mask_path) for the test set.
    """
    full_name  = DATASET_NAME
    images_tr  = os.path.join(NNUNET_RAW, full_name, "imagesTr")
    labels_tr  = os.path.join(NNUNET_RAW, full_name, "labelsTr")
    images_ts  = os.path.join(NNUNET_RAW, full_name, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    image_paths, mask_paths = build_file_lists(DATA_DIR)
    n_total = len(image_paths)
    n_test  = max(1, int(n_total * TEST_SPLIT))
    n_train = n_total - n_test

    print(f"Total samples : {n_total}")
    print(f"Train         : {n_train}")
    print(f"Test          : {n_test}")

    # deterministic split — last TEST_SPLIT fraction becomes test
    train_imgs  = image_paths[:n_train]
    train_masks = mask_paths[:n_train]
    test_imgs   = image_paths[n_train:]
    test_masks  = mask_paths[n_train:]

    # --- training set → NIfTI ---
    print("\nConverting training set to NIfTI...")
    for idx, (img_path, mask_path) in enumerate(zip(train_imgs, train_masks)):
        case_id = f"bone_{idx:04d}"

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: could not read {img_path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))
        nib.save(
            nib.Nifti1Image(img[:, :, np.newaxis].astype(np.float32), np.eye(4)),
            os.path.join(images_tr, f"{case_id}_0000.nii.gz")
        )

        mask = load_merged_mask(mask_path)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        nib.save(
            nib.Nifti1Image(mask[:, :, np.newaxis].astype(np.uint8), np.eye(4)),
            os.path.join(labels_tr, f"{case_id}.nii.gz")
        )

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{n_train} done")

    # --- test set → NIfTI (images only, GT kept separately for evaluation) ---
    print("\nConverting test set to NIfTI...")
    test_case_ids = []
    for idx, (img_path, mask_path) in enumerate(zip(test_imgs, test_masks)):
        case_id = f"bone_test_{idx:04d}"

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: could not read {img_path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))

        # save as NIfTI for nnU-Net inference
        nib.save(
            nib.Nifti1Image(img[:, :, np.newaxis].astype(np.float32), np.eye(4)),
            os.path.join(images_ts, f"{case_id}_0000.nii.gz")
        )

        # also save as PNG so visualizer can load it easily
        cv2.imwrite(
            os.path.join(RAW_DIR, full_name, "imagesTs", f"{case_id}_0000.png"),
            img
        )

        test_case_ids.append((case_id, mask_path))

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {"0": "X-Ray"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(NNUNET_RAW, full_name, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset prepared. {n_train} train, {n_test} test cases.")
    return test_case_ids


# ─────────────────────────────────────────────
#  NNUNET PIPELINE STEPS
# ─────────────────────────────────────────────

def run_nnunet_plan_and_preprocess():
    """Run nnU-Net planning and preprocessing."""
    print("\nPlanning and preprocessing...")
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity"
    ], check=True)
    print("Preprocessing done.")


def run_nnunet_train(fold=0):
    """Train a single fold."""
    print(f"\nTraining fold {fold}...")
    subprocess.run([
        "nnUNetv2_train",
        DATASET_ID,
        CONFIG,
        str(fold),
        "--npz",
        "-tr", TRAINER
    ], check=True)
    print(f"Fold {fold} done.")


def run_nnunet_predict():
    """
    Run ensemble inference on the test set using all 5 trained folds.
    Saves predicted NIfTI masks, then converts them to PNG for the visualizer.
    """
    images_ts   = os.path.join(NNUNET_RAW,  DATASET_NAME, "imagesTs")
    nifti_preds = os.path.join(BASE_DIR, "nnunet_predictions_nifti")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(nifti_preds,     exist_ok=True)

    print("\nRunning nnU-Net inference (ensemble over 5 folds)...")
    subprocess.run([
        "nnUNetv2_predict",
        "-i",  images_ts,
        "-o",  nifti_preds,
        "-d",  DATASET_ID,
        "-c",  CONFIG,
        "-tr", TRAINER,
        "-f",  "0", "1", "2", "3", "4",  # all folds → ensemble
        "--save_probabilities"
    ], check=True)

    # convert NIfTI predictions → PNG binary masks for the visualizer
    print("\nConverting NIfTI predictions to PNG...")
    for fname in os.listdir(nifti_preds):
        if not fname.endswith(".nii.gz"):
            continue

        case_id = fname.replace(".nii.gz", "")
        nifti   = nib.load(os.path.join(nifti_preds, fname))
        arr     = nifti.get_fdata()

        # nnU-Net 2D saves (H, W, 1) or (H, W) — squeeze to 2D
        arr = np.squeeze(arr)
        mask_png = ((arr > 0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(PREDICTIONS_DIR, f"{case_id}.png"), mask_png)

    print(f"Predictions saved to: {PREDICTIONS_DIR}")


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_dice(pred, mask):
    pred  = (pred > 0).astype(np.uint8)
    mask  = (mask > 0).astype(np.uint8)
    inter = (pred * mask).sum()
    return float((2 * inter + 1e-9) / (pred.sum() + mask.sum() + 1e-9))


def compute_hd95(pred, mask):
    pred = (pred > 0).astype(bool)
    mask = (mask > 0).astype(bool)

    if not pred.any() or not mask.any():
        return float(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2))

    d1 = distance_transform_edt(~mask)[pred]
    d2 = distance_transform_edt(~pred)[mask]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def collect_predictions_with_metrics(test_case_ids):
    records = []

    for case_id, gt_mask_path in test_case_ids:
        pred_path = os.path.join(PREDICTIONS_DIR, f"{case_id}.png")

        if not os.path.exists(pred_path):
            print(f"  WARNING: prediction not found for {case_id}, skipping.")
            continue

        pred     = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"  WARNING: could not read prediction for {case_id}, skipping.")
            continue
        pred     = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_bin = (pred > 127).astype(np.uint8)

        mask = load_merged_mask(gt_mask_path)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # load original image (PNG saved alongside NIfTI during prepare step)
        img_path = os.path.join(RAW_DIR, DATASET_NAME, "imagesTs", f"{case_id}_0000.png")
        img = cv2.imread(img_path)
        if img is None:
            # fallback: blank canvas so the rest of the record still works
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))

        dice = compute_dice(pred_bin, mask)
        hd95 = compute_hd95(pred_bin, mask)

        records.append({
            "case_id": case_id,
            "dice":    dice,
            "hd95":    hd95,
            "img":     img,
            "mask":    mask,
            "pred":    pred_bin,
        })

    records.sort(key=lambda x: x["dice"])  # ascending — worst first, best last
    return records


def visualize_nnunet_predictions(test_case_ids, n_best=3, n_worst=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = collect_predictions_with_metrics(test_case_ids)

    if len(records) == 0:
        print("No predictions found.")
        return

    # guard against small test sets
    n_best  = min(n_best,  len(records) // 2)
    n_worst = min(n_worst, len(records) // 2)
    n_rows  = n_best + n_worst

    worst    = records[:n_worst]
    best     = records[-n_best:][::-1]
    selected = best + worst

    labels = (
        [f"Best {i+1}  | Dice: {best[i]['dice']:.4f}"  for i in range(n_best)] +
        [f"Worst {i+1} | Dice: {worst[i]['dice']:.4f}" for i in range(n_worst)]
    )

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing consistent
    plt.suptitle("nnUNet — Bone Segmentation: Global Best & Worst Predictions",
                 fontsize=14, fontweight="bold")

    for row, (rec, label) in enumerate(zip(selected, labels)):
        axes[row, 0].imshow(rec["img"], cmap="gray")
        axes[row, 0].set_title(f"{label} | HD95: {rec['hd95']:.1f}px", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rec["img"], cmap="gray")
        axes[row, 1].imshow(rec["mask"], alpha=0.5, cmap="Reds")
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(rec["img"], cmap="gray")
        axes[row, 2].imshow(rec["pred"], alpha=0.5, cmap="Blues")
        axes[row, 2].set_title("nnUNet Prediction", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUTPUT_DIR, "predictions_visualization_nnunet.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # metrics histograms
    all_dices = [r["dice"] for r in records]
    all_hd95s = [r["hd95"] for r in records
                 if r["hd95"] != float("inf") and not np.isnan(r["hd95"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(all_dices, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(np.mean(all_dices), color="red", linestyle="--",
                label=f"Mean: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    ax1.set_xlabel("Dice Score")
    ax1.set_ylabel("Count")
    ax1.set_title("nnUNet — Dice Score Distribution")
    ax1.legend()

    if all_hd95s:
        ax2.hist(all_hd95s, bins=20, color="coral", edgecolor="black", alpha=0.8)
        ax2.axvline(np.mean(all_hd95s), color="red", linestyle="--",
                    label=f"Mean: {np.mean(all_hd95s):.2f} ± {np.std(all_hd95s):.2f}px")
        ax2.set_xlabel("HD95 (pixels)")
        ax2.set_ylabel("Count")
        ax2.set_title("nnUNet — HD95 Distribution")
        ax2.legend()

    plt.suptitle("nnUNet Test Set Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "results_chart_nnunet.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart_path}")

    print("\n=== nnUNet Results ===")
    print(f"Samples : {len(records)}")
    print(f"Dice    : {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    if all_hd95s:
        print(f"HD95    : {np.mean(all_hd95s):.2f} ± {np.std(all_hd95s):.2f} px")
    else:
        print("HD95    : N/A (all predictions empty)")


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. install nnU-Net
    install_nnunet()

    set_env()
    test_case_ids = prepare_nnunet_dataset()


    # 4. plan + preprocess
    run_nnunet_plan_and_preprocess()

    # 5. train all 5 folds
    for fold in range(5):
        print(f"\n>>> Training fold {fold} <<<")
        run_nnunet_train(fold=fold)

    # 6. predict on test set
    run_nnunet_predict()

    # 7. visualize best / worst + metrics chart
    visualize_nnunet_predictions(test_case_ids, n_best=3, n_worst=3)