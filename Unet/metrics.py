import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


# the value represents correctness of predicted mask based on real merged mask
def dice_score(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = (torch.sigmoid(pred) > 0.5).float()  # prediction is 0 OR 1
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice


# boundary error metric: ignore the worst 5% of distances, return the highest remaining distance
def hd95(pred, target):
    pred   = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()  # np doesnt work on gpu
    target = target.cpu().numpy()

    pred   = pred.squeeze()
    target = target.squeeze()

    if not pred.any() or not target.any():
        return 0.0

    # distance transform uses much less memory than cdist
    d1 = distance_transform_edt(~target.astype(bool))[pred.astype(bool)]   # pred → target
    d2 = distance_transform_edt(~pred.astype(bool))[target.astype(bool)]   # target → pred

    all_d = np.concatenate([d1, d2])
    if len(all_d) == 0:  # edge case: indexing produced empty arrays
        return 0.0

    return float(np.percentile(all_d, 95))