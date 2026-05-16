import os
import gc
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold  # provides train/val index splits into 5-fold
import albumentations as Augment           # augmentation lib
from albumentations.pytorch import ToTensorV2

# graph & ui
from tqdm import tqdm
import matplotlib.pyplot as plt

# helper files
from dataset import BoneSegDataset, build_file_lists
from unet import get_model
from losses import bce_dice_loss
from metrics import dice_score, hd95

#########################################################################################################
#! Constants

LEARNING_RATE = 1e-4
EPOCH         = 100   # to match with default nn-Unet step value
BATCH_SIZE    = 8

#! Directories

# LOCAL DIRS
# DATA_DIR   = "../Data"
# OUTPUT_DIR = "unet_outputs"

# KAGGLE DIRS
DATA_DIR   = "/kaggle/input/datasets/okancannazli/bone-seg-2/20250401 - ACB - 141"
OUTPUT_DIR = "/kaggle/working/outputs"

#########################################################################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CACHE = os.path.join(OUTPUT_DIR, "results_cache.json")  # persists fold results across sessions

# fold saver for interruption
def save_results(results):
    with open(RESULTS_CACHE, "w") as f:
        json.dump(results, f)


def load_results():
    if os.path.exists(RESULTS_CACHE):
        with open(RESULTS_CACHE) as f:
            return json.load(f)
    return []




def main():

    image_paths, mask_paths = build_file_lists(DATA_DIR)  # 141 matched image-mask pairs

    #!
    # 141 samples split into 5 non-overlapping folds
    # each sample appears in val set exactly once
    # fold sizes: 4x28 + 1x29 = 141 (last fold gets the remainder)
    kf = KFold(n_splits=5, shuffle=True, random_state=66)
    # each fold: all 141 samples, ~113 train / ~28 val


    # augmentation & normalize
    train_transform = Augment.Compose([
        Augment.HorizontalFlip(p=0.5),
        Augment.RandomRotate90(p=0.5),
        Augment.ShiftScaleRotate(p=0.3),
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # only normalize
    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    

    results = load_results() # resume: load already-finished folds
    finished_folds = {r["fold"] for r in results}# set of completed fold numbers

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):


        if (fold + 1) in finished_folds: # skip already-completed folds
            print(f"\n--- Fold {fold+1}/5 skipped (already done) ---")
            continue



        print(f"\n--- Fold {fold+1}/5 ---")

        train_images = [image_paths[i] for i in train_idx]  # each image
        train_masks  = [mask_paths[i]  for i in train_idx]  # that image's mask
        val_images   = [image_paths[i] for i in val_idx]
        val_masks    = [mask_paths[i]  for i in val_idx]

        train_dataset = BoneSegDataset(image_paths=train_images, mask_paths=train_masks, transform=train_transform)
        val_dataset   = BoneSegDataset(image_paths=val_images,   mask_paths=val_masks,   transform=val_transform)

        # batches into groups of 8 for training, 1 for validation
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False)

        # prioritize gpu usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = get_model().to(device)

        if torch.cuda.device_count() > 1:  # for kaggle's 2x gpu — splits each batch across both GPUs
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # weight_decay to prevent overfitting
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)  # cosine lr decay (fast → slow)

        best_dice  = 0.0
        best_path  = os.path.join(OUTPUT_DIR, f"fold{fold+1}_best.pth")

        # for visualization
        train_losses        = []
        val_dices_per_epoch = []

        start_epoch = 0

        for epoch in range(start_epoch, EPOCH):
            # --- TRAINING ---
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} Train"):
                images = images.to(device)
                masks  = masks.to(device)

                optimizer.zero_grad()
                preds = model(images)
                preds = preds.squeeze(1) # (B,1,H,W) → (B,H,W)
                loss  = bce_dice_loss(preds, masks) # loss calculation func
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # --- VALIDATION ---
            model.eval()
            val_dices = []

            with torch.no_grad():  # dont calculate grad for validation phase
                for images, masks in val_loader:
                    images = images.to(device)
                    masks  = masks.to(device)
                    preds  = model(images)
                    preds  = preds.squeeze(1)  # (B,1,H,W) → (B,H,W)
                    val_dices.append(dice_score(preds, masks).item())

            mean_dice  = np.mean(val_dices)
            epoch_loss = train_loss / len(train_loader)

            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Dice: {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                
                # DataParallel wraps model in .module — unwrap before saving so checkpoint is always clean
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state, best_path)
                print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

            train_losses.append(epoch_loss)
            val_dices_per_epoch.append(mean_dice)
            scheduler.step()



        # Loss/Dice graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, label="Train Loss")
        ax1.set_title(f"Fold {fold+1} - Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(val_dices_per_epoch, label="Val Dice", color="green")
        ax2.set_title(f"Fold {fold+1} - Dice Score")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fold{fold+1}_metrics.png"), dpi=300)
        plt.close()

        # load best model checkpoint for HD95 evaluation otherwise we always assume the last epoch is the best one
        
        # re-wrap with DataParallel if needed so inference runs on both GPUs
        raw_model = get_model().to(device)
        raw_model.load_state_dict(torch.load(best_path, map_location=device))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(raw_model)
        else:
            model = raw_model
            
        model.eval()  # disable dropout & batchnorm training behavior for inference

        fold_hd95s = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                preds  = model(images)
                preds  = preds.squeeze(1)  # (B,1,H,W) → (B,H,W)
                fold_hd95s.append(hd95(preds, masks))

        mean_hd95 = np.mean(fold_hd95s)
        print(f"Fold {fold+1} HD95: {mean_hd95:.2f}px")

        # free memory at end of fold
        del model
        torch.cuda.empty_cache()
        gc.collect()

   
        results.append({"fold": fold + 1, "best_dice": best_dice, "mean_hd95": mean_hd95})
        save_results(results)  # persist immediately so a crash doesnt lose this fold


    print("\n=== RESULTS ===")
    for r in results:
        print(f"Fold {r['fold']}: Dice={r['best_dice']:.4f}, HD95={r['mean_hd95']:.2f}px")

    dices = [r["best_dice"] for r in results]
    hd95s = [r["mean_hd95"] for r in results]
    print(f"\nOverall: Dice={np.mean(dices):.4f} ± {np.std(dices):.4f}, HD95={np.mean(hd95s):.2f} ± {np.std(hd95s):.2f}px")

    # inference
    from inference import visualize_predictions, save_results_chart
    save_results_chart(OUTPUT_DIR, results)
    visualize_predictions(OUTPUT_DIR, DATA_DIR)  # re-runs inference per fold since models were freed during training


if __name__ == "__main__":
    main()