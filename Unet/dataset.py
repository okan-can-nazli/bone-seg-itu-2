import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class BoneSegDataset(Dataset):

    # image_paths → ["/path/img1.jpg", "/path/img2.jpg", ...] — file
    # mask_paths  → ["/path/img1.json", "/path/img2.json", ...] — file
    # transform provides augmentation
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # get & set image
        image = cv2.imread(self.image_paths[idx])  # format: (H,W,3), BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
        image = cv2.resize(image, (512, 512))

        # get & set mask
        mask = load_merged_mask(self.mask_paths[idx])  # one X-ray → multiple polygon masks → merge into single binary mask
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # mask MUST contain only 0 OR 1

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"].float()
            mask  = augmented["mask"].float()

        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # numpy (H,W,3) → tensor (3,H,W), normalize (0,1)
            mask  = torch.from_numpy(mask).float()  # numpy (H,W) → tensor (H,W)

        return image, mask


########################################################################################################################################


def load_merged_mask(json_path):
    with open(json_path) as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]

    merged = np.zeros((h, w), dtype=np.uint8)  # blank mask

    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)  # polygon vertices
        cv2.fillPoly(merged, [points], 1)  # fill polygon with 1 (bone)

    return merged


def build_file_lists(data_dir):

    image_paths = []
    mask_paths  = []

    # sorted() provides consistent ordering
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            base = os.path.splitext(f)[0]
            json_file = os.path.join(data_dir, base + ".json")

            if os.path.exists(json_file):  # only add if matching json exists
                image_paths.append(os.path.join(data_dir, f))
                mask_paths.append(json_file)

    print(f"[Dataset] {len(image_paths)} matched pairs found.")
    return image_paths, mask_paths