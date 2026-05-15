import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold # provides train/val index splits into 5-fold
import albumentations as Augment # augmentation lib
from albumentations.pytorch import ToTensorV2 # convert np array into tensor

# graph & uı
from tqdm import tqdm # training progress bar 
import matplotlib.pyplot as plt

# helper files
from dataset import BoneSegDataset, build_file_lists
from unet import get_model
from losses import bce_dice_loss
from metrics import dice_score, hd95

#########################################################################################################
#! Constants

LEARNING_RATE = 1e-4
EPOCH = 50
BATCH_SIZE = 8

#! Directorys

# LOCAL DIRS
IMAGE_DIR = "../Data/images"
MASK_DIR = "../Data/masks"
OUTPUT_DIR = "unet_outputs"

#KAGGLE DIRS
# IMAGE_DIR = "/kaggle/input/datasets/okancannazli/bones-seg/New_Labels-20260504T191710Z-3-001/New_Labels"
# MASK_DIR  = "/kaggle/input/datasets/okancannazli/bones-seg/New_masks-20260504T191902Z-3-001/New_masks"
# OUTPUT_DIR = "/kaggle/working/outputs"
#########################################################################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    

    image_paths, mask_folders = build_file_lists(IMAGE_DIR, MASK_DIR)  # 499 matched image-mask pairs

    kf = KFold(n_splits=5, shuffle=True, random_state=66) 
    # each fold: all 499 samples, ~400 train / ~100 val, different split each time, select random val and train sample EVERY FOLD 
    # Fold 1: 1-100 val, 101-499 train
    # Fold 2: 101-200 val, 1-100 + 201-499 train
    # ...
    
    
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

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):  # 400,100
        print(f"\n--- Fold {fold+1}/5 ---")

        # len = 400
        train_images = [image_paths[i] for i in train_idx] # each image
        train_masks  = [mask_folders[i] for i in train_idx] # that image's mask fileS
        
        # len = 100
        val_images   = [image_paths[i] for i in val_idx]
        val_masks    = [mask_folders[i] for i in val_idx]

        train_dataset = BoneSegDataset(image_paths=train_images, mask_folders=train_masks, transform=train_transform)
        val_dataset   = BoneSegDataset(image_paths=val_images, mask_folders=val_masks, transform=val_transform)

        # batches into groups of 8 for training/validation
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

        # prioritize gpu usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = get_model().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # weight_decay to prevent overfitting (one feature focus)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6) # reduce Learning rate at every epoch based on cosine func (min = 1e-6) (fast - mid - slow change rate)

        best_dice = 0.0
        best_path = os.path.join(OUTPUT_DIR, f"fold{fold+1}_best.pth")
        
        #for visualation
        train_losses = []
        val_dices_per_epoch = []

        for epoch in range(EPOCH):
            
            # --- TRAİNİNG ---
            train_loss = 0.0
            model.train()

            
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} Train"):
                images  = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                preds = model(images)
                preds = preds.squeeze(1)  # (B,1,H,W) → (B,H,W)
                loss  = bce_dice_loss(preds, masks) # bce dice loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            

            # --- VALIDATİON ---
            model.eval()
            val_dices = []
            # val_hd95s = [] 

            with torch.no_grad(): # dont calculate grad for validation phase
                for images, masks in val_loader:
                    images  = images.to(device)
                    masks = masks.to(device)
                    preds = model(images)
                    preds = preds.squeeze(1)  # (B,1,H,W) → (B,H,W)
                    
                    val_dices.append(dice_score(preds, masks).item())
                    # val_hd95s.append(hd95(preds, masks)) # skipped: make the system slower

            mean_dice = np.mean(val_dices)
            # mean_hd95 = np.mean(val_hd95s)

            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Dice: {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model saved (Dice={best_dice:.4f})")
                
            train_losses.append(train_loss / len(train_loader))
            val_dices_per_epoch.append(mean_dice)
            scheduler.step()
            
            
        #visualation
        
        # Loss/Dice graph (mat plot lib)
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
        
        
        model.load_state_dict(torch.load(best_path)) # load best model checkpoint for HD95 evaluation (not the last epoch)
        model.eval()
        fold_hd95s = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                preds = preds.squeeze(1)  # (B,1,H,W) → (B,H,W)
                fold_hd95s.append(hd95(preds, masks))
        mean_hd95 = np.mean(fold_hd95s)
        print(f"Fold {fold+1} HD95: {mean_hd95:.2f}px")
        
        # free memory enf of the fold
        del model
        torch.cuda.empty_cache()
        
        import gc
        gc.collect() # garbage collector

        results.append({"fold": fold+1, "best_dice": best_dice, "mean_hd95": mean_hd95})



    print("\n=== RESULTS ===")
    for r in results:
        print(f"Fold {r['fold']}: Dice={r['best_dice']:.4f}, HD95={r['mean_hd95']:.2f}px")

    dices = [r["best_dice"] for r in results]
    hd95s = [r["mean_hd95"] for r in results]
    print(f"\nOverall: Dice={np.mean(dices):.4f} ± {np.std(dices):.4f}, HD95={np.mean(hd95s):.2f} ± {np.std(hd95s):.2f}px")



    # inference
    best_fold_idx  = max(range(len(results)), key=lambda i: results[i]['best_dice'])
    worst_fold_idx = min(range(len(results)), key=lambda i: results[i]['best_dice'])

    from inference import visualize_predictions, save_results_chart
    save_results_chart(OUTPUT_DIR, results)
    visualize_predictions(OUTPUT_DIR, IMAGE_DIR, MASK_DIR, best_fold_idx, worst_fold_idx) # here re-calculate outputs of the given folds cause we delete each fold after executed because of performance reasons

if __name__ == "__main__":
    main()