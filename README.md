<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.jpg"
      >
    </a>
  </p>
</div>

# Autodistill: GroundedSAM Base Model

This repository contains the code implementing [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) as a Base Model for use with [`autodistill`](https://github.com/autodistill/autodistill).

GroundedSAM combines [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) with the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to identify and segment objects in an image given text captions.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [GroundedSAM Autodistill documentation](https://autodistill.github.io/autodistill/base_models/groundedsam/).

> [!TIP]
> You can use Autodistill Grounded SAM on your own hardware using the instructions below, or use the [Roboflow hosted version of Autodistill](https://blog.roboflow.com/launch-auto-label/) to label images in the cloud.

## Installation

To use the GroundedSAM Base Model, simply install it along with a Target Model supporting the `detection` task:

```bash
pip3 install autodistill-grounded-sam autodistill-yolov8
```

You can find a full list of `detection` Target Models on [the main autodistill repo](https://github.com/autodistill/autodistill).

## Quickstart

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "shipping container": "shipping container",
        }
    )
)

# run inference on a single image
results = base_model.predict("logistics.jpeg")

plot(
    image=cv2.imread("logistics.jpeg"),
    classes=base_model.ontology.classes(),
    detections=results
)
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
