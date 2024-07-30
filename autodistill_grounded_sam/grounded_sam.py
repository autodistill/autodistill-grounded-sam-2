import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.use_deterministic_algorithms(False)

from typing import Any

import numpy as np
import supervision as sv
from autodistill_grounded_sam.helpers import (combine_detections,
                                              load_grounding_dino,
                                              load_SAM)
from autodistill.helpers import load_image
from groundingdino.util.inference import Model

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill_florence_2 import Florence2

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GroundedSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    florence_2_predictor: Florence2
    sam_2_predictor: SamPredictor
    box_threshold: float
    text_threshold: float

    def __init__(
        self, ontology: CaptionOntology
    ):
        self.ontology = ontology
        self.grounding_dino_model = Florence2(ontology=ontology)
        self.sam_predictor = load_SAM()

    def predict(self, input: Any) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        florence_2_detections = self.florence_2_predictor.predict(image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(image)
            result_masks = []
            for box in florence_2_detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(box=box, multimask_output=False)
                index = np.argmax(scores)
                result_masks.append(masks[index])

            florence_2_detections.mask = np.array(result_masks)
        return florence_2_detections
