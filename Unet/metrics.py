import torch
#from scipy.spatial.distance import cdist
import numpy as np
from scipy.ndimage import distance_transform_edt


# the value represents correctness of predicted mask based on real merged mask
def dice_score(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = (torch.sigmoid(pred) > 0.5).float()  # prediction is 0 OR 1
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice 


# boundry error metric: ignore the worst %5 point of distances after that return the highest point distance
def hd95(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float().cpu().numpy() # np doesnt work on gpu
    target = target.cpu().numpy()
    
    
    pred = pred.squeeze()
    target = target.squeeze()

    if not pred.any() or not target.any():
        return 0.0

    # distance transform uses much less memory than cdist
    d1 = distance_transform_edt(~target.astype(bool))[pred.astype(bool)]  # pred → target
    d2 = distance_transform_edt(~pred.astype(bool))[target.astype(bool)]  # target → pred
    
    
    

    #! MEMORY OVERLOAD CODE
    # # list of bones coordinates
    # pred_points = np.argwhere(pred == 1)
    # target_points = np.argwhere(target == 1)

    # if len(pred_points) == 0 or len(target_points) == 0:
    #     return 0.0

    # all_distances = cdist(pred_points, target_points) # distances between each pred_point and target_point
    # d1 = all_distances.min(axis=1)  # pred → target: unexisted-bone prediction
    # d2 = all_distances.min(axis=0)  # target → pred: missed real bones

    return float(np.percentile(np.concatenate([d1, d2]), 95))