import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil 

def convert_voc_xml(image_info, annotations, categories, output_dir, sub_dir):
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    image_id = image_info['id']

    image_folder = os.path.join("coco", sub_dir, "val2025")
    img_path = os.path.join(image_folder, filename)
    shutil.copy(src=img_path, dst=output_dir)

    annotation = ET.Element('Annotation')

    ET.SubElement(annotation, 'folder').text = 'VOC2025'
    ET.SubElement(annotation, 'filename').text = filename

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

    for ann in annotations:
        if ann['image_id'] == image_id:
            category_id = ann['category_id']
            bboxes = ann['bbox']
            # print(f'bboxes :{bbox}')
            for i  in range(len(category_id)):
                category_name = categories[category_id[i]]
                bbox = bboxes[i]
                # print(f'name: {category_name}, bbox: {bbox}')
            
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[0] + bbox[2])
                ymax = int(bbox[1] + bbox[3])

                object_elem = ET.SubElement(annotation, 'object')
                ET.SubElement(object_elem, 'name').text = category_name
                # ET.SubElement(object_elem, 'pose').text = 'Unspecified'
                # ET.SubElement(object_elem, 'truncated').text = '0'
                ET.SubElement(object_elem, 'difficult').text = '0'

                bndbox = ET.SubElement(object_elem, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(xmin)
                ET.SubElement(bndbox, 'ymin').text = str(ymin)
                ET.SubElement(bndbox, 'xmax').text = str(xmax)
                ET.SubElement(bndbox, 'ymax').text = str(ymax)

    xml_str = ET.tostring(annotation, encoding='utf-8').decode()
    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xml')
    with open(output_path, 'w') as f:
        f.write(xml_str)

    print(f"Saved: {output_path}")


def coco2voc(json_path, output_dir):
    with open(json_path) as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f'categories: {categories}')
    sub_dir = json_path.split('/')[1]
    output_dirs = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dirs, exist_ok=True)

    for image_info in tqdm(images, desc="Converting COCO to VOC"):
        convert_voc_xml(image_info, annotations, categories, output_dirs, sub_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert COCO JSON to VOC XML")
    parser.add_argument("--json_path", default="coco", type=str, help="Path to COCO JSON file containing directories")
    parser.add_argument("--output_dir", default="VOC2025", type=str, help="Directory to save VOC XML files with images")
    args = parser.parse_args()

    for subdir in os.listdir(args.json_path):
        json_file = os.path.join(args.json_path, subdir,"annotations", "person_keypoints_val2025_kpts.json")
        coco2voc(json_file, args.output_dir)


