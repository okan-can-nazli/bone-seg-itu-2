# BCE + Dice combined.
# BCE alone: model cheats by predicting all background (class imbalance).
# Dice alone: unstable early in training.
# Together: BCE stabilizes, Dice forces the model to actually find bones.

import torch
import torch.nn as nn

# (1-dice_score) loss calculation based on overall image (explorer)
def dice_loss(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = torch.sigmoid(pred)  # logits → probabilities (0-1)
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return 1 - dice 

# loss calculation based on pixel by pixel (stabilizer)
def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target.float())
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice