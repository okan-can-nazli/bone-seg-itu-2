import os
import json
import numpy as np
import cv2
import nibabel as nib

from Unet.dataset import build_file_lists, load_merged_mask

#########################################################################################################
#! Constants

DATASET_ID   = "001"
DATASET_NAME = "BoneSeg"

#! Directories

# LOCAL DIRS
# DATA_DIR   = "../Data"
# OUTPUT_DIR = "nnunet_raw"

# KAGGLE DIRS
DATA_DIR   = "/kaggle/input/datasets/foxcancoy/bone-seg-2/Data"
OUTPUT_DIR = "/kaggle/working/nnunet_raw"
#########################################################################################################


def main():

    full_name  = f"Dataset{DATASET_ID}_{DATASET_NAME}"
    images_out = os.path.join(OUTPUT_DIR, full_name, "imagesTr")
    labels_out = os.path.join(OUTPUT_DIR, full_name, "labelsTr")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_paths, mask_paths = build_file_lists(DATA_DIR) 
    
    print(f"Converting {len(image_paths)} samples to NIfTI...")

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        case_id = f"bone_{idx:04d}"

        # image: grayscale → (H, W, 1) NIfTI
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))
        
        # nnU-net only use .nii.gz files
        nib.save(
            nib.Nifti1Image(img[:, :, np.newaxis].astype(np.float32), np.eye(4)),
            os.path.join(images_out, f"{case_id}_0000.nii.gz")
        )

        # mask: merged binary → (H, W, 1) NIfTI
        mask = load_merged_mask(mask_path)  # all polygons → single binary mask
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        nib.save(
            nib.Nifti1Image(mask[:, :, np.newaxis].astype(np.uint8), np.eye(4)),
            os.path.join(labels_out, f"{case_id}.nii.gz")
        )

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(image_paths)} done")

    # nnU-Net reads this file to understand the dataset before training
    # it uses this info to auto-configure network architecture and hyperparameters
    dataset_json = {
        "channel_names": {"0": "X-Ray"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": len(image_paths),
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(OUTPUT_DIR, full_name, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDone. {len(image_paths)} samples saved to: {OUTPUT_DIR}/{full_name}")


if __name__ == "__main__":
    main()