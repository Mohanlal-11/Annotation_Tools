import argparse
import datetime
import json
import os
from copy import deepcopy

import cv2
import numpy as np

from annotator import BboxAnnotator
from bbox_utils import increment_idx, load_annotations, upload_annotations, authenticate_drive

LEFT_ARM_COLOR = (216, 235, 52, 255)
LEFT_LEG_COLOR = (235, 107, 52, 255)
LEFT_SIDE_COLOR = (245, 188, 113, 255)
LEFT_FACE_COLOR = (235, 52, 107, 255)

RIGHT_ARM_COLOR = (52, 235, 216, 255)
RIGHT_LEG_COLOR = (52, 107, 235, 255)
RIGHT_SIDE_COLOR = (52, 171, 235, 255)
RIGHT_FACE_COLOR = (107, 52, 235, 255)
coco_markers = [
    # ["nose", cv2.MARKER_CROSS, NEUTRAL_COLOR],
    # ["left_eye", cv2.MARKER_SQUARE, LEFT_FACE_COLOR],
    # ["right_eye", cv2.MARKER_SQUARE, RIGHT_FACE_COLOR],
    ["head", cv2.MARKER_TRIANGLE_UP, LEFT_FACE_COLOR],
    ["neck", cv2.MARKER_SQUARE, LEFT_FACE_COLOR],
    # ["right_ear", cv2.MARKER_CROSS, RIGHT_FACE_COLOR],
    ["left_shoulder", cv2.MARKER_TRIANGLE_UP, LEFT_ARM_COLOR],
    ["right_shoulder", cv2.MARKER_TRIANGLE_UP, RIGHT_ARM_COLOR],
    ["left_elbow", cv2.MARKER_SQUARE, LEFT_ARM_COLOR],
    ["right_elbow", cv2.MARKER_SQUARE, RIGHT_ARM_COLOR],
    ["left_wrist", cv2.MARKER_TILTED_CROSS, LEFT_ARM_COLOR],
    ["right_wrist", cv2.MARKER_TILTED_CROSS, RIGHT_ARM_COLOR],
    ["left_hip", cv2.MARKER_TRIANGLE_UP, LEFT_LEG_COLOR],
    ["right_hip", cv2.MARKER_TRIANGLE_UP, RIGHT_LEG_COLOR],
    ["left_knee", cv2.MARKER_SQUARE, LEFT_LEG_COLOR],
    ["right_knee", cv2.MARKER_SQUARE, RIGHT_LEG_COLOR],
    ["left_ankle", cv2.MARKER_TILTED_CROSS, LEFT_LEG_COLOR],
    ["right_ankle", cv2.MARKER_TILTED_CROSS, RIGHT_LEG_COLOR],
    ["middle", cv2.MARKER_TILTED_CROSS, RIGHT_LEG_COLOR],
]

class_path = "./classes.txt"
def list_classes():
    with open(class_path, "r") as fp:
        datas = fp.readlines()
    classes = []
    for cls in datas:
        classes.append(cls.strip('\n'))
        
    return classes

CLASSES = list_classes()
CLASSES.insert(0, "nothing")

def save_annotations(image_idx, annotations_file, annotations, ann_dict, update_date=False):
    annotations["annotations"] = []
    for ann_list in ann_dict.values():
        annotations["annotations"].extend(ann_list)
        

    if update_date:
        annotations["info"]["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coco_folder",
        type=str,
        help="Folder containing the dataset for annotation",
    )
    parser.add_argument("--img-path", type=str, help="Path to the folder with images", default=None)
    parser.add_argument("--cloud-upload", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cloud-folder", type=str, help="Google Drive folder ID for uploading annotations", default='root')

    args = parser.parse_args()

    assert os.path.exists(args.coco_folder), "Given folder ({:s}) not found".format(
        args.coco_folder
    )
    assert os.path.isdir(args.coco_folder), "Given folder ({:s}) is not a folder".format(
        args.coco_folder
    )

    if args.img_path is None:
        subdirs = [
            d
            for d in os.listdir(args.coco_folder)
            if os.path.isdir(os.path.join(args.coco_folder, d))
        ]
        if "val2025" in subdirs:
            args.img_path = os.path.join(args.coco_folder, "val2025")
        elif "images" in subdirs:
            args.img_path = os.path.join(args.coco_folder, "images")
        else:
            args.img_path = args.coco_folder

    args = prepare_filestructure(args)

    return args


def prepare_filestructure(args):
    if args.coco_folder == args.img_path:
        # Create a new folder for the images and annotations
        new_img_path = os.path.join(args.coco_folder, "val2025")
        os.makedirs(new_img_path, exist_ok=True)

        # Move all images to the new folder
        for img in os.listdir(args.img_path):
            os.rename(os.path.join(args.img_path, img), os.path.join(new_img_path, img))
        args.img_path = new_img_path

    # Create a new folder for the annotations
    new_ann_path = os.path.join(args.coco_folder, "annotations")
    os.makedirs(new_ann_path, exist_ok=True)
    return args


def get_ann_filepath(args):
    ann_filename = os.path.join(args.coco_folder, "annotations", "person_keypoints_val2025_kpts.json")
    if not (os.path.exists(ann_filename) and os.path.isfile(ann_filename)):
        create_ann_file(ann_filename, args.img_path)

    return ann_filename


def create_ann_file(ann_filename, img_path):
    ann_dict = {
        "info": {
            "year": datetime.datetime.now().strftime("%Y"),
            "version": 1.0,
            "description": "COCO-like dataset created at Nepal",
            "author": "Mohan Lal Shrestha",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        },
        "images": [],
        "categories": [],
        "annotations":[]
    }

    for img_i, img_name in enumerate(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, img_name))
        ann_dict["images"].append(
            {
                "id": img_i,
                "file_name": img_name,
                "width": img.shape[1],
                "height": img.shape[0],
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            }
        )
        ann_dict["annotations"].append({
            "image_id": img_i,
            "bbox": [],
            "category_id": [],
            "id": np.random.randint(low=0, high=1e6)
        })
        
    for i in range(len(CLASSES)):
        ann_dict["categories"].append({
                "supercategory": "None",
                "id": i,
                "name": CLASSES[i]
            })
        
    with open(ann_filename, "w") as f:
        json.dump(ann_dict, f, indent=2)

        
def main(args):
    # Load the data
    json_file_path = get_ann_filepath(args)
    coco_data, _, _, ann_dict = load_annotations(json_file_path)
    # print(f'ana dict: {ann_dict}')
    # print(f'coco: {coco_data}')
    img_list = [(img["file_name"], img["id"]) for img in coco_data["images"]]
    img_idx = 0
    save_path = os.path.join(args.coco_folder, "annotations", "person_keypoints_val2025_kpts.json")
    if args.cloud_upload:
        from pydrive.drive import GoogleDrive
        gauth = authenticate_drive()
        drive = GoogleDrive(gauth)
        folder_id = args.cloud_folder
        file_name = os.path.basename(save_path)

    cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
    # print(f'initial ann: {ann_dict[img_list[img_idx][1]]}')

    ia = BboxAnnotator(
        ann_dict[img_list[img_idx][1]],
        img_list[img_idx][1],
        os.path.join(args.img_path, img_list[img_idx][0]),
        is_start=img_idx == 0,
    )
    cv2.setMouseCallback("Image", ia.mouse_callback)
    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = cv2.waitKey(100)
        if k == ord("m") or k == 83:  # toggle current image
            # if cls_id == 1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            img_idx = increment_idx(img_idx, len(img_list), 1)
            # print(f'ann_dict: {ann_dict}')
            # print(f'in next: {ann_dict}')
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # print(f'ann dicr: {ann_dict}')
            # save_annotations(img_idx-1, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)
            # generate_graph(json_file_path, img_idx-1)

            cv2.setMouseCallback("Image", ia.mouse_callback)
            
        # elif k == ord("c"):
        #     if cls_id < len(CLASSES)-1:
        #         cls_id += 1
        #         # if img_idx == 0:
        #         ann_dict[img_list[img_idx][1]] = ia.get_annotation(cls_id)
                
        #         ia = BboxAnnotator(
        #             box_idx,
        #             cls_id,
        #             ann_dict[img_list[img_idx][1]],
        #             img_list[img_idx][1],
        #             os.path.join(args.img_path, img_list[img_idx][0]),
        #             is_start=img_idx == 0,
        #         )
                
        #         save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id, cls_indx=cat_idx)
        #         cv2.setMouseCallback("Image", ia.mouse_callback)
        #     else:
        #         cls_id = 0
                
            
        elif k == ord("n") or k == 81:
            # if cls_id == 1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            img_idx = increment_idx(img_idx, len(img_list), -1)
            # print(f'in previous key: {ann_dict}')
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord(",") or k == 83:  # toggle current image
            # if cls_id ==1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            img_idx = increment_idx(img_idx, len(img_list), -10)
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # save_annotations(img_idx+10, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord(".") or k == 81:
            # if cls_id == 1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            img_idx = increment_idx(img_idx, len(img_list), 10)
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # save_annotations(img_idx-10, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("x"):
            # if cls_id == 1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            img_idx = np.random.randint(len(img_list))
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # save_annotations(current_idx, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("q"):
            # cls_id = extract_label(save_path, img_idx, cat_id=cat_idx)
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            break
        elif k == ord("u"):
            # if cls_id == 1:
            ann_dict[img_list[img_idx][1]] = ia.get_annotation()
            save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
            while not ann_dict[img_list[img_idx][1]] == []:
                img_idx = increment_idx(img_idx, len(img_list), 1)
            ia = BboxAnnotator(
                ann_dict[img_list[img_idx][1]],
                img_list[img_idx][1],
                os.path.join(args.img_path, img_list[img_idx][0]),
                is_start=img_idx == 0,
            )
            # save_annotations(current_idx, save_path, coco_data, ann_dict, update_date=True, cls_id=cls_id)

            cv2.setMouseCallback("Image", ia.mouse_callback)

        else:
            ia.key_pressed(k)
    cv2.destroyAllWindows()
    save_annotations(img_idx, save_path, coco_data, ann_dict, update_date=True)
    
    if args.cloud_upload:
        upload_annotations(drive, coco_data, file_name, folder_id)

if __name__ == "__main__":
    args = parse_args()
    main(args)
