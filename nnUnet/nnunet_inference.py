"""
nnunet_inference.py
-------------------
nnUNet setup, prediction, and best/worst visualization.
Mirrors the same output style as inference.py (UNet).

Kaggle Usage:
    1. Run the setup cell to install nnUNet and prepare folder structure
    2. Run nnUNetv2_predict to generate predictions
    3. Call visualize_nnunet_predictions() to produce the visualization PNG

Expected folder structure after nnUNet predict:
    /kaggle/working/
    ├── nnUNet_raw/
    │   └── Dataset001_Bone/
    │       ├── imagesTr/   ← training images  (_0000.png)
    │       ├── labelsTr/   ← training masks   (.png)
    │       ├── imagesTs/   ← test images       (_0000.png)
    │       └── dataset.json
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
    └── nnUNet_predictions/  ← nnUNet writes predicted masks here
"""

import os
import json
import shutil
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold

from dataset import build_file_lists, load_merged_mask


# ─────────────────────────────────────────────
#  KAGGLE DIRS  (edit if needed)
# ─────────────────────────────────────────────
DATA_DIR        = "/kaggle/input/datasets/okancannazli/bone-seg-2/20250401 - ACB - 141"
BASE_DIR        = "/kaggle/working"
RAW_DIR         = os.path.join(BASE_DIR, "nnUNet_raw")
PREPROCESSED    = os.path.join(BASE_DIR, "nnUNet_preprocessed")
RESULTS_DIR     = os.path.join(BASE_DIR, "nnUNet_results")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "nnUNet_predictions")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
DATASET_ID      = "001"
DATASET_NAME    = f"Dataset{DATASET_ID}_Bone"


# ─────────────────────────────────────────────
#  STEP 1 — Install nnUNet (run once)
# ─────────────────────────────────────────────

def install_nnunet():
    """Install nnUNetv2. Run this in a Kaggle cell before anything else."""
    subprocess.run(["pip", "install", "nnunetv2", "-q"], check=True)
    print("nnUNetv2 installed.")


# ─────────────────────────────────────────────
#  STEP 2 — Prepare dataset in nnUNet format
# ─────────────────────────────────────────────

def prepare_nnunet_dataset():
    """
    Convert image/json pairs → nnUNet Dataset001_Bone folder structure.
    Images → PNG with suffix _0000 (nnUNet channel convention)
    Masks  → binary PNG (0 = background, 1 = bone)
    """
    image_paths, mask_paths = build_file_lists(DATA_DIR)

    # 80/20 split for imagesTr / imagesTs
    n = len(image_paths)
    n_train = int(n * 0.8)
    train_imgs, train_masks = image_paths[:n_train], mask_paths[:n_train]
    test_imgs,  test_masks  = image_paths[n_train:], mask_paths[n_train:]

    dataset_dir = os.path.join(RAW_DIR, DATASET_NAME)
    images_tr   = os.path.join(dataset_dir, "imagesTr")
    labels_tr   = os.path.join(dataset_dir, "labelsTr")
    images_ts   = os.path.join(dataset_dir, "imagesTs")

    for d in [images_tr, labels_tr, images_ts]:
        os.makedirs(d, exist_ok=True)

    # --- training set ---
    for i, (img_p, mask_p) in enumerate(zip(train_imgs, train_masks)):
        case_id = f"bone_{i:04d}"

        # image: RGB → grayscale, save as _0000.png
        img = cv2.imread(img_p)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(images_tr, f"{case_id}_0000.png"), img)

        # mask: binary 0/1 → save as PNG (nnUNet reads label value directly)
        mask = load_merged_mask(mask_p)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(labels_tr, f"{case_id}.png"), mask)

    # --- test set (images only, no labels needed for predict) ---
    test_case_ids = []
    for i, img_p in enumerate(test_imgs):
        case_id = f"bone_test_{i:04d}"
        img = cv2.imread(img_p)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(images_ts, f"{case_id}_0000.png"), img)
        test_case_ids.append((case_id, test_masks[i]))  # keep mask path for evaluation

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {"0": "X-ray"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": len(train_imgs),
        "file_ending": ".png"
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset prepared: {len(train_imgs)} train, {len(test_imgs)} test")
    return test_case_ids


# ─────────────────────────────────────────────
#  STEP 3 — Set env vars & run nnUNet commands
# ─────────────────────────────────────────────

def set_env():
    os.environ["nnUNet_raw"]         = RAW_DIR
    os.environ["nnUNet_preprocessed"] = PREPROCESSED
    os.environ["nnUNet_results"]     = RESULTS_DIR
    print("nnUNet env vars set.")


def run_nnunet_plan_and_preprocess():
    set_env()
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity"
    ], check=True)


def run_nnunet_train(fold: int = 0, trainer: str = "nnUNetTrainer"):
    """Train one fold. Run for fold 0–4 for full 5-fold CV."""
    set_env()
    subprocess.run([
        "nnUNetv2_train",
        DATASET_ID, "2d", str(fold),
        "--tr", trainer
    ], check=True)


def run_nnunet_predict():
    """Run inference on imagesTs using all 5 folds (ensemble)."""
    set_env()
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    subprocess.run([
        "nnUNetv2_predict",
        "-i",  os.path.join(RAW_DIR, DATASET_NAME, "imagesTs"),
        "-o",  PREDICTIONS_DIR,
        "-d",  DATASET_ID,
        "-c",  "2d",
        "-f",  "0", "1", "2", "3", "4",   # all 5 folds
        "--save_probabilities"
    ], check=True)
    print(f"Predictions saved to: {PREDICTIONS_DIR}")


# ─────────────────────────────────────────────
#  STEP 4 — Evaluate & visualize best/worst
# ─────────────────────────────────────────────

def compute_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    pred  = (pred > 0).astype(np.uint8)
    mask  = (mask > 0).astype(np.uint8)
    inter = (pred * mask).sum()
    return float((2 * inter + 1e-9) / (pred.sum() + mask.sum() + 1e-9))


def compute_hd95(pred: np.ndarray, mask: np.ndarray) -> float:
    """95th percentile Hausdorff distance (in pixels)."""
    from scipy.spatial.distance import directed_hausdorff
    pred_pts = np.argwhere((pred > 0).astype(np.uint8))
    mask_pts = np.argwhere((mask > 0).astype(np.uint8))
    if len(pred_pts) == 0 or len(mask_pts) == 0:
        return float("inf")
    d1 = np.array([np.min(np.linalg.norm(mask_pts - p, axis=1)) for p in pred_pts])
    d2 = np.array([np.min(np.linalg.norm(pred_pts - p, axis=1)) for p in mask_pts])
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def collect_predictions_with_metrics(test_case_ids):
    """
    Load predicted masks from PREDICTIONS_DIR, compute Dice + HD95
    for each test sample. Returns list sorted by Dice ascending.

    test_case_ids: list of (case_id, gt_mask_json_path) from prepare_nnunet_dataset()
    """
    records = []

    for case_id, gt_mask_path in test_case_ids:
        pred_path = os.path.join(PREDICTIONS_DIR, f"{case_id}.png")
        if not os.path.exists(pred_path):
            print(f"  WARNING: prediction not found for {case_id}, skipping.")
            continue

        # load prediction (binary)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_bin = (pred > 127).astype(np.uint8)

        # load ground truth
        mask = load_merged_mask(gt_mask_path)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # load original image for visualization
        # image is in imagesTs as case_id_0000.png
        img_path = os.path.join(RAW_DIR, DATASET_NAME, "imagesTs", f"{case_id}_0000.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        dice = compute_dice(pred_bin, mask)
        hd   = compute_hd95(pred_bin, mask)

        records.append({
            "case_id": case_id,
            "dice":    dice,
            "hd95":    hd,
            "img":     img,
            "mask":    mask,
            "pred":    pred_bin,
        })

    records.sort(key=lambda x: x["dice"])
    return records


def visualize_nnunet_predictions(test_case_ids, n_best=3, n_worst=3):
    """
    Produces predictions_visualization_nnunet.png  (same layout as UNet version)
    and results_chart_nnunet.png with overall Dice / HD95 bar chart.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = collect_predictions_with_metrics(test_case_ids)

    if len(records) == 0:
        print("No predictions found. Did you run run_nnunet_predict()?")
        return

    worst = records[:n_worst]
    best  = records[-n_best:][::-1]
    selected = best + worst
    labels = (
        [f"Best {i+1}" for i in range(n_best)] +
        [f"Worst {i+1}" for i in range(n_worst)]
    )

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 4))
    plt.suptitle("nnUNet — Bone Segmentation: Best & Worst Predictions", fontsize=14, fontweight="bold")

    for row, (rec, label) in enumerate(zip(selected, labels)):
        img      = rec["img"]
        mask     = rec["mask"]
        pred_bin = rec["pred"]
        dice     = rec["dice"]
        hd       = rec["hd95"]

        # col 0: original image
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"{label} | Dice: {dice:.4f} | HD95: {hd:.1f}px", fontsize=9)
        axes[row, 0].axis("off")

        # col 1: ground truth overlay
        axes[row, 1].imshow(img, cmap="gray")
        axes[row, 1].imshow(mask, alpha=0.5, cmap="Reds")
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        # col 2: prediction overlay
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(pred_bin, alpha=0.5, cmap="Blues")
        axes[row, 2].set_title("nnUNet Prediction", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUTPUT_DIR, "predictions_visualization_nnunet.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # ── summary bar chart ──────────────────────────────────────────────
    all_dices = [r["dice"] for r in records]
    all_hd95s = [r["hd95"] for r in records if r["hd95"] != float("inf")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(all_dices, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(np.mean(all_dices), color="red", linestyle="--",
                label=f"Mean: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    ax1.set_xlabel("Dice Score")
    ax1.set_ylabel("Count")
    ax1.set_title("nnUNet — Dice Score Distribution")
    ax1.legend()

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

    # ── print summary ──────────────────────────────────────────────────
    print("\n=== nnUNet Results ===")
    print(f"Samples evaluated : {len(records)}")
    print(f"Dice  : {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    print(f"HD95  : {np.mean(all_hd95s):.2f} ± {np.std(all_hd95s):.2f} px")


# ─────────────────────────────────────────────
#  FULL PIPELINE  (run top to bottom in Kaggle)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. install
    install_nnunet()

    # 2. prepare data & get test case references
    test_case_ids = prepare_nnunet_dataset()

    # 3. set env vars
    set_env()

    # 4. plan + preprocess
    run_nnunet_plan_and_preprocess()

    # 5. train all 5 folds  (takes ~hours on GPU, each fold separately)
    for fold in range(5):
        print(f"\n>>> Training fold {fold} <<<")
        run_nnunet_train(fold=fold)

    # 6. predict on test set
    run_nnunet_predict()

    # 7. visualize best / worst + metrics chart
    visualize_nnunet_predictions(test_case_ids, n_best=3, n_worst=3)
