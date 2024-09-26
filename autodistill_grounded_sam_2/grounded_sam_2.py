import copy
import json
import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from autodistill_florence_2 import Florence2
from PIL import Image

from autodistill_grounded_sam_2.helpers import (
    combine_detections,
    load_grounding_dino,
    load_SAM,
)
from autodistill_grounded_sam_2.utils.mask_dictionary_model import (
    MaskDictionaryModel,
    ObjectInfo,
)
from autodistill_grounded_sam_2.utils.utils import (
    convert_int64,
    create_sv_detections,
    extract_frames,
    get_dir_names,
    get_frames_for_sam,
    make_temp_dir,
    remove_temp_dir,
)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SamPredictor = load_SAM()

SUPPORTED_GROUNDING_MODELS = ["Florence 2", "Grounding DINO"]


@dataclass
class GroundedSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    box_threshold: float
    text_threshold: float
    video_mode: bool = False

    def __init__(
        self,
        ontology: CaptionOntology,
        model="Florence 2",
        grounding_dino_box_threshold=0.35,
        grounding_dino_text_threshold=0.25,
        video_mode=False,
    ):
        if model not in SUPPORTED_GROUNDING_MODELS:
            raise ValueError(
                f"Grounding model {model} is not supported. Supported models are {SUPPORTED_GROUNDING_MODELS}"
            )
        if self.video_mode:
            assert (
                model == "Grounding DINO"
            ), "Video mode only supports Grounding DINO model"
        self.ontology = ontology
        if model == "Florence 2":
            self.florence_2_predictor = Florence2(ontology=ontology)
        elif model == "Grounding DINO":
            self.grounding_dino_model = load_grounding_dino()
        self.sam_2_predictor = SamPredictor
        self.model = model
        self.grounding_dino_box_threshold = grounding_dino_box_threshold
        self.grounding_dino_text_threshold = grounding_dino_text_threshold
        self.video_mode = video_mode

    def _predict_image(self, input: str) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        if self.model == "Florence 2":
            detections = self.florence_2_predictor.predict(image)
        elif self.model == "Grounding DINO":
            # GroundingDINO predictions
            detections_list = []

            for i, description in enumerate(self.ontology.prompts()):
                # detect objects
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[description],
                    box_threshold=self.grounding_dino_box_threshold,
                    text_threshold=self.grounding_dino_text_threshold,
                )

                detections_list.append(detections)

            detections = combine_detections(
                detections_list, overwrite_class_ids=range(len(detections_list))
            )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(image)
            result_masks = []
            for box in detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(
                    box=box, multimask_output=False
                )
                index = np.argmax(scores)
                masks = masks.astype(bool)
                result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        return detections

    def _predict_video(self, input: str) -> sv.Detections:
        # parts of the video code is from https://github.com/IDEA-Research/Grounded-SAM-2/tree/main
        _, mask_data_dir, json_data_dir, result_dir, video_dir = get_dir_names()
        extract_frames(input, video_dir)
        frame_names = get_frames_for_sam(video_dir)
        image_predictor, video_predictor = load_SAM()
        inference_state = video_predictor.init_state(video_path=video_dir)
        step = 5

        sam2_masks = MaskDictionaryModel()
        grounding_dino_model = load_grounding_dino()
        PROMPT_TYPE_FOR_VIDEO = "mask"
        objects_count = 0
        make_temp_dir()
        for start_frame_idx in range(0, len(frame_names), step):
            img_path = os.path.join(video_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_dino = load_image(image, return_format="cv2")

            image_base_name = frame_names[start_frame_idx].split(".")[0]
            mask_dict = MaskDictionaryModel(
                promote_type=PROMPT_TYPE_FOR_VIDEO,
                mask_name=f"mask_{image_base_name}.npy",
            )

            ontology = CaptionOntology(
                {
                    "person": "person",
                    "shipping container": "shipping container",
                }
            )

            # GroundingDINO predictions
            detections_list = []

            for i, description in enumerate(ontology.prompts()):
                # detect objects
                detections = grounding_dino_model.predict_with_classes(
                    image=image_dino,
                    classes=[description],
                    box_threshold=self.grounding_dino_box_threshold,
                    text_threshold=self.grounding_dino_text_threshold,
                )

                detections_list.append(detections)

            objects = combine_detections(
                detections_list, overwrite_class_ids=range(len(detections_list))
            )

            input_boxes = []
            confidences = []
            class_names = []

            for idx, obj in enumerate(objects):
                input_boxes.append(obj[0])
                confidences.append(obj[2])
                class_names.append(obj[3])

            input_boxes = np.array(input_boxes)

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(DEVICE),
                    box_list=torch.tensor(input_boxes),
                    label_list=class_names,
                )
            else:
                raise NotImplementedError(
                    "SAM 2 video predictor only support mask prompts"
                )

            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks,
                iou_threshold=0.8,
                objects_count=objects_count,
            )
            video_predictor.reset_state(inference_state)

            print("objects_count", objects_count)
            if len(mask_dict.labels) == 0:
                print(
                    "No object detected in the frame, skip the frame {}".format(
                        start_frame_idx
                    )
                )
                continue

            for object_id, object_info in mask_dict.labels.items():
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )

            video_segments = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(
                inference_state,
                max_frame_num_to_track=step,
                start_frame_idx=start_frame_idx,
            ):
                frame_masks = MaskDictionaryModel()

                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = out_mask_logits[i] > 0.0  # .cpu().numpy()
                    object_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id),
                    )
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            for _, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(
                    frame_masks_info.mask_height, frame_masks_info.mask_width
                )
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(
                    os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img
                )

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(
                    json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json")
                )
                # Convert the JSON data
                converted_data = convert_int64(json_data)

                with open(json_data_path, "w") as f:
                    json.dump(converted_data, f)

        detections, labels = create_sv_detections(
            video_dir, mask_data_dir, json_data_dir, result_dir
        )
        remove_temp_dir()
        return detections

    def predict(self, input: str) -> sv.Detections:
        if self.video_mode:
            return self._predict_video(input)
        else:
            return self._predict_image(input)
