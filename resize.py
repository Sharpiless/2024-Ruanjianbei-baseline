import os
import json
import cv2
from tqdm import tqdm

def resize_image(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    return cv2.resize(image, (width, height)), width, height

def update_annotations(annotations, ratio):
    for annotation in annotations:
        annotation['bbox'] = [coord * ratio for coord in annotation['bbox']]
        if 'segmentation' in annotation:
            annotation['segmentation'] = [
                [coord * ratio for coord in segment] for segment in annotation['segmentation']
            ]
    return annotations

def resize_dataset(image_dir, json_file, output_dir, ratio):
    with open(json_file) as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    resized_image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(resized_image_dir):
        os.makedirs(resized_image_dir)

    for image_info in tqdm(images):
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        resized_image, new_width, new_height = resize_image(image, ratio)

        image_info['width'] = new_width
        image_info['height'] = new_height
        cv2.imwrite(os.path.join(resized_image_dir, image_info['file_name']), resized_image)

    data['annotations'] = update_annotations(annotations, ratio)

    with open(os.path.join(output_dir, os.path.basename(json_file)), 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
image_dir = '24dog/image'
train_json = '24dog/train.json'
val_json = '24dog/val.json'
output_dir = '24dog_resized'
resize_ratio = 0.5  # Change this to your desired ratio

resize_dataset(image_dir, train_json, output_dir, resize_ratio)
resize_dataset(image_dir, val_json, output_dir, resize_ratio)
