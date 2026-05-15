# nnU-Net 2D — 5-fold cross validation (50 epochs)
# Kaggle notebook: https://www.kaggle.com/code/okancannazli/nnu-net-bone-seg

import os
import subprocess

#########################################################################################################
#! Constants

DATASET_ID = "001"
CONFIG     = "2d"
TRAINER    = "nnUNetTrainer_50epochs"  # built-in 50-epoch trainer

#! Directories

# LOCAL DIRS
# NNUNET_RAW          = "nnunet_raw"
# NNUNET_PREPROCESSED = "nnunet_preprocessed"
# NNUNET_RESULTS      = "nnunet_results"

# KAGGLE DIRS
NNUNET_RAW          = "/kaggle/working/nnunet_raw"
NNUNET_PREPROCESSED = "/kaggle/working/nnunet_preprocessed"
NNUNET_RESULTS      = "/kaggle/working/nnunet_results"
#########################################################################################################

os.environ["nnUNet_raw"]          = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
os.environ["nnUNet_results"]      = NNUNET_RESULTS

os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)
os.makedirs(NNUNET_RESULTS,      exist_ok=True)


def main():

    print("Planning and preprocessing...")
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity"
    ], check=True)

    for fold in range(5):
        print(f"\nTraining fold {fold}...")
        subprocess.run([
            "nnUNetv2_train",
            DATASET_ID,
            CONFIG,
            str(fold),
            "--npz",
            "-tr", TRAINER
        ], check=True)

    print("\nAll folds done.")


if __name__ == "__main__":
    main()
