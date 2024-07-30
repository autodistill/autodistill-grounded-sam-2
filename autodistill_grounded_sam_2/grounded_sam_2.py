import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from autodistill_florence_2 import Florence2

from autodistill_grounded_sam_2.helpers import load_SAM

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SamPredictor = load_SAM()


@dataclass
class GroundedSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    florence_2_predictor: Florence2
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.florence_2_predictor = Florence2(ontology=ontology)
        self.sam_2_predictor = SamPredictor

    def predict(self, input: Any) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        florence_2_detections = self.florence_2_predictor.predict(image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(image)
            result_masks = []
            for box in florence_2_detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(
                    box=box, multimask_output=False
                )
                index = np.argmax(scores)
                masks = masks.astype(bool)
                result_masks.append(masks[index])

            florence_2_detections.mask = np.array(result_masks)

        return florence_2_detections
