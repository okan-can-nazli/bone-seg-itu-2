import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as Augment
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

from dataset import build_file_lists, load_merged_mask
from unet import get_model


def get_predictions(output_dir, image_dir, mask_dir, fold_idx):
    """
    Given a fold index, returns sorted list of (idx, dice, img, mask, pred_bin)
    """
    image_paths, mask_folders = build_file_lists(image_dir, mask_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=66)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # get fold's data set
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        if fold == fold_idx:
            val_images = [image_paths[i] for i in val_idx]
            val_masks  = [mask_folders[i] for i in val_idx]
            break

    # reload the model
    model = get_model().to(device)
    model.load_state_dict(torch.load(
        os.path.join(output_dir, f"fold{fold_idx+1}_best.pth"),
        map_location=device))
    model.eval()

    sample_dices = []
    with torch.no_grad():
        for idx in range(len(val_images)):
            img = cv2.imread(val_images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (512, 512))

            mask = load_merged_mask(val_masks[idx])
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            aug = val_transform(image=img_resized)
            img_tensor = aug["image"].unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_tensor).squeeze()).cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            intersection = (pred_bin * mask).sum()
            dice = (2 * intersection + 1e-9) / (pred_bin.sum() + mask.sum() + 1e-9)
            sample_dices.append((idx, float(dice), img_resized, mask, pred_bin))

    del model
    torch.cuda.empty_cache()

    sample_dices.sort(key=lambda x: x[1])
    return sample_dices


def visualize_predictions(output_dir, image_dir, mask_dir, best_fold_idx, worst_fold_idx):
    os.makedirs(output_dir, exist_ok=True)

    # best of 3 from best fold
    best_preds  = get_predictions(output_dir, image_dir, mask_dir, best_fold_idx)
    best_3      = best_preds[-3:][::-1]  # en iyiden başla

    # worst of 3 from worst fold
    worst_preds = get_predictions(output_dir, image_dir, mask_dir, worst_fold_idx)
    worst_3     = worst_preds[:3]

    selected = best_3 + worst_3
    labels   = [
        f'Best 1 (Fold {best_fold_idx+1})',
        f'Best 2 (Fold {best_fold_idx+1})',
        f'Best 3 (Fold {best_fold_idx+1})',
        f'Worst 1 (Fold {worst_fold_idx+1})',
        f'Worst 2 (Fold {worst_fold_idx+1})',
        f'Worst 3 (Fold {worst_fold_idx+1})',
    ]

    fig, axes = plt.subplots(6, 3, figsize=(12, 24))
    plt.suptitle("Bone Segmentation: Best & Worst Predictions", fontsize=14)

    for row, (idx, dice, img_resized, mask, pred_bin) in enumerate(selected):
        axes[row, 0].imshow(img_resized, cmap='gray')
        axes[row, 0].set_title(f"{labels[row]} | Dice: {dice:.4f}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_resized, cmap='gray')
        axes[row, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

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

    ax1.bar(folds, dices, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(np.mean(dices), color='red', linestyle='--', label=f'Mean: {np.mean(dices):.4f}')
    for i, v in enumerate(dices):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylim([0.88, 1.0])
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score per Fold')
    ax1.legend()

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
    
    #MANUEL TESTİNG OF PNG OUTPUTS
    
    # LOCAL DIRS
    IMAGE_DIR = "Data/images"
    MASK_DIR = "Data/masks"
    OUTPUT_DIR = "outputs"

    #KAGGLE DIRS
    # IMAGE_DIR = "/kaggle/input/datasets/okancannazli/bones-seg/New_Labels-20260504T191710Z-3-001/New_Labels"
    # MASK_DIR  = "/kaggle/input/datasets/okancannazli/bones-seg/New_masks-20260504T191902Z-3-001/New_masks"
    # OUTPUT_DIR = "/kaggle/working/outputs"

    best_fold_idx = 4
    worst_fold_idx = 1
    
    visualize_predictions(OUTPUT_DIR, IMAGE_DIR, MASK_DIR, best_fold_idx=best_fold_idx, worst_fold_idx=worst_fold_idx)