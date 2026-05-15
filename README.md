# Bone Segmentation in X-Ray Images

Binary semantic segmentation of bones in X-ray images using U-Net with a pretrained ResNet34 encoder. Also includes an nnU-Net 2D baseline for comparison.

## Results

### U-Net + ResNet34

| Fold | Dice ↑ | HD95 ↓ (px) |
|------|--------|-------------|
| 1    | 0.9418 | 3.52        |
| 2    | 0.9425 | 2.67        |
| 3    | 0.9368 | 8.36        |
| 4    | 0.9424 | 2.83        |
| 5    | 0.9324 | 4.63        |
| **Mean** | **0.9392 ± 0.0040** | **4.40 ± 2.09** |

### nnU-Net 2D (50 epochs)

| Fold | Dice ↑ |
|------|--------|
| 1    | 0.9452 |
| 2    | 0.9410 |
| 3    | 0.9402 |
| 4    | 0.9308 |
| 5    | 0.9369 |
| **Mean** | **0.9388** |

## Architecture

- **Model:** U-Net with ResNet34 encoder (ImageNet pretrained)
- **Library:** segmentation-models-pytorch
- **Input:** 3-channel RGB, 512×512 px
- **Output:** 1-channel binary mask (bone=1, background=0)
- **Parameters:** 24,436,369 trainable

## Dataset

- 499 X-ray images (500 total, 1 skipped — patient 283 contains .gif)
- Multiple per-bone .npy masks merged into a single binary mask at runtime
- Image formats: .jpg, .jpeg, .png

## Training

- **Loss:** BCE + Dice combined (0.5 weight each)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=50, eta_min=1e-6)
- **Epochs:** 50 per fold
- **Batch size:** 8 (train) / 1 (validation)
- **Augmentations:** HorizontalFlip, RandomRotate90, ShiftScaleRotate
- **Platform:** Kaggle — GPU T4 x2

## Project Structure

```
bone-seg-itu/
├──Unet/
    ├── dataset.py           # Dataset class + file list builder
    ├── unet.py              # U-Net model (segmentation-models-pytorch)
    ├── losses.py            # BCE + Dice combined loss
    ├── metrics.py           # Dice score + HD95 (Hausdorff Distance)
    ├── cross_validation.py  # 5-fold CV training — main script
    ├── inference.py         # Best & worst prediction visualization
├── nnUNet/
    ├── dataset_nnunet.py  # Converts data to nnU-Net NIfTI format
    └── nnunet.py          # nnU-Net 2D training pipeline
```

## Kaggle Notebooks

- **U-Net:** https://www.kaggle.com/code/okancannazli/bone-segmentation
- **nnU-Net:** https://www.kaggle.com/code/okancannazli/nnu-net-bone-seg

## How to Run

### U-Net on Kaggle

```bash
!git clone https://github.com/okan-can-nazli/bone-seg-itu.git
%cd bone-seg-itu
!pip install segmentation-models-pytorch
!python cross_validation.py
```

### nnU-Net on Kaggle

```bash
!git clone https://github.com/okan-can-nazli/bone-seg-itu.git
%cd bone-seg-itu
!pip install nnunetv2
!python nnUNet/dataset_nnunet.py   # convert data
!python nnUNet/nnunet.py           # train all folds
```

### Local

Update directory paths in `cross_validation.py` or `nnUNet/dataset_nnunet.py`, then run.

## Key Design Decisions

- **Pretrained encoder:** ImageNet weights on ResNet34 enable fast convergence on medical images
- **BCE+Dice loss:** BCE stabilizes early training, Dice forces accurate overlap
- **HD95 over HD100:** 95th percentile is more robust to boundary outliers
- **val batch_size=1:** Ensures accurate per-sample Dice and HD95 computation
- **nnU-Net format:** Data converted to NIfTI (.nii.gz) with shape (H, W, 1) for 2D nnU-Net compatibility
