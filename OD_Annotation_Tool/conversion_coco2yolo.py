import os 
import json
import sys
from tqdm import tqdm

full_path = os.path.abspath(__file__)
root = full_path[:-24]

def convert_to_yolo(image_info, annotations, categories, sub_dir):
    image_name = image_info["file_name"]
    im_name_only = image_name.split(".")[0]
    image_id = image_info["id"]
    image_width = image_info["width"]
    image_height = image_info["height"]
    
    full_image_path = os.path.join(root, "coco", sub_dir, "val2025", image_name)
    txt_path = os.path.join(root, "coco", sub_dir, "val2025", im_name_only)
    all_datas = f"{full_image_path} "
    for annotation in annotations:
        if annotation["image_id"] == image_id:
            datas = []
            boxes = annotation["bbox"]
            classes = annotation["category_id"]
            for box, label in zip(boxes,classes):
                data = f"{label} "
                xmin = box[0]
                ymin = box[1]
                w = box[2]
                h = box[3]
                
                x_c = xmin + w/2
                y_c = ymin + h/2
                
                norm_x = x_c/image_width
                norm_y = y_c/image_height
                norm_w = w/image_width
                norm_h = h/image_height
                
                data += f"{norm_x} {norm_y} {norm_w} {norm_h} "
                datas.append(data)
                all_datas += f"{label},{norm_x},{norm_y},{norm_w},{norm_h} "
            
            with open(f"{txt_path}.txt", "w") as f:
                f.write("\n".join(datas))
    return full_image_path, all_datas

def coco2yolo(json_path, output_dir):
    with open(json_path, "r") as fp:
        coco_datas = json.load(fp)

    images = coco_datas['images']
    annotations = coco_datas['annotations']
    categories = {cat['name']:cat['id'] for cat in coco_datas['categories']}
    print(f'categories: {categories}')
    sub_dir = json_path.split('/')[1]
    
    output_dirs = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dirs, exist_ok=True)
    
    all_imgs_path = []
    all_datas = []
    for image_info in tqdm(images, desc="Converting COCO to YOLO"):
        full_im_path, all_data = convert_to_yolo(image_info, annotations, categories, sub_dir)
        all_imgs_path.append(full_im_path)
        all_datas.append(all_data)
        
    with open(f"{output_dirs}/{sub_dir}_path_only.txt", "w") as fw:
        fw.write("\n".join(all_imgs_path))
    
    with open(f"{output_dirs}/{sub_dir}_all.txt", "w") as fw:
        fw.write("\n".join(all_datas))
          
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'Usage: python3 <path/to/directories/containing jsons> <output_Dir(YOLO)>')
    else:
        json_file_dir = sys.argv[1]
        output_dir = sys.argv[2]
        for subdir in os.listdir(json_file_dir):
            json_file = os.path.join(json_file_dir, subdir,"annotations", "person_keypoints_val2025_kpts.json")
            coco2yolo(json_path=json_file, output_dir=output_dir)