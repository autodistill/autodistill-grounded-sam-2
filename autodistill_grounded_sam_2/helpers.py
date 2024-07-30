import os
import subprocess
import sys
import urllib.request

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO will run very slowly.")


def load_SAM():
    cur_dir = os.getcwd()

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything_2")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam2_hiera_base_plus.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    os.chdir(SAM_CACHE_DIR)

    if not os.path.isdir("~/.cache/autodistill/segment_anything_2/segment-anything-2"):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/segment-anything-2.git",
            ]
        )

        os.chdir("segment-anything-2")

        subprocess.run(["pip", "install", "-e", "."])

    sys.path.append("~/.cache/autodistill/segment_anything_2/segment-anything-2")

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "~/.cache/autodistill/segment_anything_2/sam2_hiera_base_plus.pth"
    checkpoint = os.path.expanduser(checkpoint)
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    os.chdir(cur_dir)

    return predictor
