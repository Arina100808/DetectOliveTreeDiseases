from utils.datasets import download_roboflow_dataset
from utils.visualizations import display_image
import supervision as sv
from PIL import Image
import yaml
import glob
import os

dataset, dataset_dir = download_roboflow_dataset("raddaoui-amal", "mon-pfe", version=2)

# Load class names
data_yaml_path = os.path.join(dataset.location, "data.yaml")
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']

# Create an output directory
output_dir = dataset_dir + "_detection"
os.makedirs(output_dir, exist_ok=True)

splits = ['train', 'valid', 'test']
for split in splits:
    labels_path = os.path.join(dataset_dir, split, "labels")
    images_path = os.path.join(dataset_dir, split, "images")

    if not os.path.exists(labels_path):
        continue  # skip if split doesn't exist

    # Create corresponding output folders
    output_labels_path = os.path.join(output_dir, split, "labels")
    output_images_path = os.path.join(output_dir, split, "images")
    os.makedirs(output_labels_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)

    # Copy images to a new directory
    for image_file in glob.glob(os.path.join(images_path, "*.*")):
        os.system(f'cp "{image_file}" "{output_images_path}/"')

    for label_file in glob.glob(os.path.join(labels_path, "*.txt")):
        image_filename = os.path.basename(label_file).replace('.txt', '.jpg')  # or .png
        image_file = os.path.join(images_path, image_filename)

        if not os.path.exists(image_file):
            print(f"Image file {image_file} not found, skipping")
            continue

        with Image.open(image_file) as img:
            width, height = img.size

        with open(label_file, 'r') as f:
            lines = f.readlines()

        output_lines = []

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox_normalized = list(map(float, parts[1:5]))
            polygon_normalized = list(map(float, parts[5:]))

            polygon = []
            for i in range(0, len(polygon_normalized), 2):
                x = polygon_normalized[i] * width
                y = polygon_normalized[i + 1] * height
                polygon.append((x, y))

            x_min, y_min, x_max, y_max = sv.polygon_to_xyxy(polygon)

            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            output_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
            output_lines.append(output_line)

        # Write a new object detection label file
        output_label_file = os.path.join(output_labels_path, os.path.basename(label_file))
        with open(output_label_file, 'w') as f_out:
            f_out.writelines(output_lines)

# Save class names into a YOLO-compatible data.yaml
data_yaml = {
    'names': class_names,
    'nc': len(class_names)  # number of classes
}

with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
    yaml.dump(data_yaml, f)

display_image(output_dir, class_names)

# Create a zip file
# shutil.make_archive(output_dir, 'zip', output_dir)
