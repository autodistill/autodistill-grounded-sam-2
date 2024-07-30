import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Model(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        pass

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        pass