# Code partly from from https://github.com/IDEA-Research/Grounded-SAM-2/tree/main


import json
import os
import shutil

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for saving the video
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (width, height)
    )

    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def draw_masks_and_box_with_supervision(
    raw_image_path, mask_path, json_path, output_path
):
    raw_image_name_list = os.listdir(raw_image_path)
    raw_image_name_list.sort()
    for raw_image_name in raw_image_name_list:
        image_path = os.path.join(raw_image_path, raw_image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image file not found.")
        # load mask
        mask_npy_path = os.path.join(
            mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy"
        )
        mask = np.load(mask_npy_path)
        # color map
        unique_ids = np.unique(mask)

        # get each mask from unique mask file
        all_object_masks = []
        for uid in unique_ids:
            if uid == 0:  # skip background id
                continue
            else:
                object_mask = mask == uid
                all_object_masks.append(object_mask[None])

        if len(all_object_masks) == 0:
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, image)
            continue
        # get n masks: (n, h, w)
        all_object_masks = np.concatenate(all_object_masks, axis=0)

        # load box information
        file_path = os.path.join(
            json_path, "mask_" + raw_image_name.split(".")[0] + ".json"
        )

        all_object_boxes = []
        all_object_ids = []
        all_class_names = []
        object_id_to_name = {}
        with open(file_path, "r") as file:
            json_data = json.load(file)
            for obj_id, obj_item in json_data["labels"].items():
                # box id
                instance_id = obj_item["instance_id"]
                if instance_id not in unique_ids:  # not a valid box
                    continue
                # box coordinates
                x1, y1, x2, y2 = (
                    obj_item["x1"],
                    obj_item["y1"],
                    obj_item["x2"],
                    obj_item["y2"],
                )
                all_object_boxes.append([x1, y1, x2, y2])
                # box name
                class_name = obj_item["class_name"]

                # build id list and id2name mapping
                all_object_ids.append(instance_id)
                all_class_names.append(class_name)
                object_id_to_name[instance_id] = class_name

        # Adjust object id and boxes to ascending order
        paired_id_and_box = zip(all_object_ids, all_object_boxes)
        sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])

        # Because we get the mask data as ascending order, so we also need to ascend box and ids
        all_object_ids = [pair[0] for pair in sorted_pair]
        all_object_boxes = [pair[1] for pair in sorted_pair]

        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )

        # custom label to show both id and class name
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_class_names)
        ]

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        output_image_path = os.path.join(output_path, raw_image_name)
        cv2.imwrite(output_image_path, annotated_frame)
        print(f"Annotated image saved as {output_image_path}")


def create_sv_detections(raw_image_path, mask_path, json_path, output_path):
    raw_image_name_list = os.listdir(raw_image_path)
    raw_image_name_list.sort()
    for raw_image_name in raw_image_name_list:
        image_path = os.path.join(raw_image_path, raw_image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image file not found.")
        # load mask
        mask_npy_path = os.path.join(
            mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy"
        )
        mask = np.load(mask_npy_path)
        # color map
        unique_ids = np.unique(mask)

        # get each mask from unique mask file
        all_object_masks = []
        for uid in unique_ids:
            if uid == 0:  # skip background id
                continue
            else:
                object_mask = mask == uid
                all_object_masks.append(object_mask[None])

        if len(all_object_masks) == 0:
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, image)
            continue
        # get n masks: (n, h, w)
        all_object_masks = np.concatenate(all_object_masks, axis=0)

        # load box information
        file_path = os.path.join(
            json_path, "mask_" + raw_image_name.split(".")[0] + ".json"
        )

        all_object_boxes = []
        all_object_ids = []
        all_class_names = []
        object_id_to_name = {}
        with open(file_path, "r") as file:
            json_data = json.load(file)
            for obj_id, obj_item in json_data["labels"].items():
                # box id
                instance_id = obj_item["instance_id"]
                if instance_id not in unique_ids:  # not a valid box
                    continue
                # box coordinates
                x1, y1, x2, y2 = (
                    obj_item["x1"],
                    obj_item["y1"],
                    obj_item["x2"],
                    obj_item["y2"],
                )
                all_object_boxes.append([x1, y1, x2, y2])
                # box name
                class_name = obj_item["class_name"]

                # build id list and id2name mapping
                all_object_ids.append(instance_id)
                all_class_names.append(class_name)
                object_id_to_name[instance_id] = class_name

        # Adjust object id and boxes to ascending order
        paired_id_and_box = zip(all_object_ids, all_object_boxes)
        sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])

        # Because we get the mask data as ascending order, so we also need to ascend box and ids
        all_object_ids = [pair[0] for pair in sorted_pair]
        all_object_boxes = [pair[1] for pair in sorted_pair]

        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )

        # custom label to show both id and class name
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_class_names)
        ]

        return detections, labels


def convert_int64(obj):
    if isinstance(obj, dict):
        return {k: convert_int64(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def extract_frames(video_name, video_dir):
    # Create a directory to store the outputs
    os.makedirs(video_dir, exist_ok=True)
    # Extract the frames from the video
    cap = cv2.VideoCapture(video_name)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(video_dir, f"{i}.png"), frame)
        i += 1
    cap.release()


def get_dir_names():
    output_dir = "./outputs"
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    video_dir = "videos"
    return output_dir, mask_data_dir, json_data_dir, result_dir, video_dir


def make_temp_dir():
    output_dir, mask_data_dir, json_data_dir, result_dir, video_dir = get_dir_names()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(mask_data_dir, exist_ok=True)
    os.makedirs(json_data_dir, exist_ok=True)


def remove_temp_dir():
    output_dir, mask_data_dir, json_data_dir, result_dir, video_dir = get_dir_names()

    def rempve_dir(dir_path):
        # Check if the directory exists
        if os.path.exists(dir_path):
            # Remove the directory and its contents
            shutil.rmtree(dir_path)

        # Optionally, recreate the directory if needed
        os.makedirs(dir_path)

    # Remove the directories
    removing_dir = [output_dir, video_dir, result_dir, mask_data_dir, json_data_dir]
    for dir_path in removing_dir:
        rempve_dir(dir_path)


def get_frames_for_sam(video_dir):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names
