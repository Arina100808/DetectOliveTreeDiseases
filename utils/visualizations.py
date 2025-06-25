import matplotlib.pyplot as plt
import cv2
import os

def display_image(folder_path, class_names):
    # Display an example image with a converted label
    # Select one example
    output_images_path = os.path.join(folder_path, "train", "images")
    output_labels_path = os.path.join(folder_path, "train", "labels")
    example_image_path = os.path.join(output_images_path, os.listdir(output_images_path)[0])
    example_label_path = os.path.join(output_labels_path,
                                      os.path.splitext(os.path.basename(example_image_path))[0] + '.txt')

    # Load image
    image = cv2.imread(example_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Read bounding box
    with open(example_label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id, x_center, y_center, bbox_width, bbox_height = parts

        # Convert from YOLO normalized format to absolute coordinates
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Add class label text
        label = class_names[int(class_id)]
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
