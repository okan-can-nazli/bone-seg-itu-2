import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as Augment # augmentation lib
from albumentations.pytorch import ToTensorV2 # convert np array into tensor
from sklearn.model_selection import KFold # provides train/val index splits into 5-fold

from dataset import build_file_lists, load_merged_mask
from unet import get_model


def get_predictions(output_dir, data_dir, fold_idx):

    image_paths, mask_paths = build_file_lists(data_dir) # all matched image-mask pairs

    kf = KFold(n_splits=5, shuffle=True, random_state=66) # same seed as training — reproduces exact same splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # only normalize, no augmentation for inference
    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # get this fold's val indices
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        if fold == fold_idx:
            val_images = [image_paths[i] for i in val_idx]
            val_masks  = [mask_paths[i] for i in val_idx]
            break

    # reload best checkpoint for this fold (not last epoch)
    model = get_model().to(device)
    model.load_state_dict(torch.load(
        os.path.join(output_dir, f"fold{fold_idx+1}_best.pth"),
        map_location=device))
    model.eval()

    sample_dices = []

    with torch.no_grad(): # no grad needed for inference
        for idx in range(len(val_images)):

            # load & preprocess image
            img = cv2.imread(val_images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (512, 512))

            # load ground truth mask
            mask = load_merged_mask(val_masks[idx])
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            # normalize & predict
            aug = val_transform(image=img_resized)
            img_tensor = aug["image"].unsqueeze(0).to(device) # (1,3,512,512)
            pred = torch.sigmoid(model(img_tensor).squeeze()).cpu().numpy() # logits → probabilities
            pred_bin = (pred > 0.5).astype(np.uint8) # probabilities → binary mask

            # inline dice for sorting (no need to import metrics)
            intersection = (pred_bin * mask).sum()
            dice = (2 * intersection + 1e-9) / (pred_bin.sum() + mask.sum() + 1e-9)

            sample_dices.append((fold_idx, idx, float(dice), img_resized, mask, pred_bin))
            # tuple: (fold_idx, sample_idx, dice, original_img, gt_mask, predicted_mask)

    # free gpu memory after each fold
    del model
    torch.cuda.empty_cache()

    return sample_dices # unsorted — global sort happens in visualize_predictions


def visualize_predictions(output_dir, data_dir, n_folds=5):

    os.makedirs(output_dir, exist_ok=True)

    # collect predictions from ALL folds
    all_preds = []
    for fold_idx in range(n_folds):
        print(f"  Loading predictions for fold {fold_idx+1}...")
        all_preds.extend(get_predictions(output_dir, data_dir, fold_idx))

    # global sort by dice score across all folds
    all_preds.sort(key=lambda x: x[2])

    worst_3 = all_preds[:3]          # lowest dice globally
    best_3  = all_preds[-3:][::-1]   # highest dice globally
    selected = best_3 + worst_3

    labels = [
        f'Best 1  (Fold {best_3[0][0]+1}) | Dice: {best_3[0][2]:.4f}',
        f'Best 2  (Fold {best_3[1][0]+1}) | Dice: {best_3[1][2]:.4f}',
        f'Best 3  (Fold {best_3[2][0]+1}) | Dice: {best_3[2][2]:.4f}',
        f'Worst 1 (Fold {worst_3[0][0]+1}) | Dice: {worst_3[0][2]:.4f}',
        f'Worst 2 (Fold {worst_3[1][0]+1}) | Dice: {worst_3[1][2]:.4f}',
        f'Worst 3 (Fold {worst_3[2][0]+1}) | Dice: {worst_3[2][2]:.4f}',
    ]

    # 6 rows (best3 + worst3) x 3 cols (image | ground truth | prediction)
    fig, axes = plt.subplots(6, 3, figsize=(12, 24))
    plt.suptitle("Bone Segmentation: Global Best & Worst Predictions", fontsize=14)

    for row, (fold_idx, idx, dice, img_resized, mask, pred_bin) in enumerate(selected):

        # col 0: original image
        axes[row, 0].imshow(img_resized, cmap='gray')
        axes[row, 0].set_title(labels[row], fontsize=8)
        axes[row, 0].axis('off')

        # col 1: ground truth overlay (red)
        axes[row, 1].imshow(img_resized, cmap='gray')
        axes[row, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        # col 2: prediction overlay (blue)
        axes[row, 2].imshow(img_resized, cmap='gray')
        axes[row, 2].imshow(pred_bin, alpha=0.5, cmap='Blues')
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, "predictions_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: predictions_visualization.png")


def save_results_chart(output_dir, results):

    folds = [f"Fold {r['fold']}" for r in results]
    dices = [r['best_dice'] for r in results]
    hd95s = [r['mean_hd95'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # dice bar chart
    ax1.bar(folds, dices, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(np.mean(dices), color='red', linestyle='--', label=f'Mean: {np.mean(dices):.4f}')
    for i, v in enumerate(dices):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylim([max(0.0, min(dices) - 0.05), 1.0])
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score per Fold')
    ax1.legend()

    # hd95 bar chart
    ax2.bar(folds, hd95s, color='coral', alpha=0.8, edgecolor='black')
    ax2.axhline(np.mean(hd95s), color='red', linestyle='--', label=f'Mean: {np.mean(hd95s):.2f}px')
    for i, v in enumerate(hd95s):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('HD95 (pixels)')
    ax2.set_title('HD95 per Fold')
    ax2.legend()

    plt.suptitle('5-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results_chart.png")


if __name__ == "__main__":

    # KAGGLE DIRS
    DATA_DIR   = "/kaggle/input/datasets/okancannazli/bone-seg-2/20250401 - ACB - 141"
    OUTPUT_DIR = "/kaggle/working/outputs"

    # LOCAL DIRS
    # DATA_DIR   = "../Data"
    # OUTPUT_DIR = "unet_outputs"

    visualize_predictions(OUTPUT_DIR, DATA_DIR)