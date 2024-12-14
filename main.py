import os
import cv2
import numpy as np

# Folder Train_Images
input_dir = r"C:\Users\pauli\OneDrive\Bureau\PNU\4 - IA Programming\project\Dataset\train\images"
# Folder output resize Train_images
output_dir = r"C:\Users\pauli\OneDrive\Bureau\PNU\4 - IA Programming\project\Dataset\resized_train\images"
os.makedirs(output_dir, exist_ok=True)

# Folder for validation images
input_val_dir = r"C:\Users\pauli\OneDrive\Bureau\PNU\4 - IA Programming\project\Dataset\valid\images"
# Folder output resize Validation_images
output_val_dir = r"C:\Users\pauli\OneDrive\Bureau\PNU\4 - IA Programming\project\Dataset\resized_valid\images"
os.makedirs(output_val_dir, exist_ok=True)

# New size
new_size = (640, 640)

# Function to resize and increase contrast of image
def resize_and_increase_contrast(image_path, new_size, alpha=1.5, beta=0):
    # Load image with OpenCV
    image = cv2.imread(image_path)
    
    # Resize image
    resized_image = cv2.resize(image, new_size)
    
    # Increase contrast
    contrasted_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)
    
    return contrasted_image

# Function to process all images in a directory (Train and Validation)
def process_images(input_dir, output_dir, new_size, alpha=1.5, beta=0):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            # Resize and increase contrast
            contrasted_image = resize_and_increase_contrast(img_path, new_size, alpha, beta)
            
            # Save the processed image
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, contrasted_image)
    
    print(f"Processing completed for images in {output_dir}.")

# Apply processing to all training images
process_images(input_dir, output_dir, new_size)

# Apply processing to all validation images
process_images(input_val_dir, output_val_dir, new_size)

print("----- Data Preprocessing and Augmentations -------")

config_path = "config.yaml"
with open(config_path, "r") as file:
    config_data = file.read()

# Modify the paths in the config to point to resized_train
updated_config = config_data.replace(
    "C:/Users/pauli/OneDrive/Bureau/PNU/4 - IA Programming/project/Dataset/train/images",
    "C:/Users/pauli/OneDrive/Bureau/PNU/4 - IA Programming/project/Dataset/resized_train/images"
)

updated_config = updated_config.replace(
    "C:/Users/pauli/OneDrive/Bureau/PNU/4 - IA Programming/project/Dataset/valid/images",
    "C:/Users/pauli/OneDrive/Bureau/PNU/4 - IA Programming/project/Dataset/resized_valid/images"
)

with open(config_path, "w") as file:
    file.write(updated_config)

print("Paths updated in config.yaml.")

from ultralytics import YOLO
import time

# Load the model
model = YOLO("yolov8n.yaml")

# Train the model
results = model.train(
    data="config.yaml",  
    epochs=60,           
    batch=16,             
    optimizer="Adam",    # Adam optimizer
    lr0=1e-3,            # Initial learning rate
    momentum=0.937,      # Momentum of SGD
    weight_decay=0.0005, # Weight decay to prevent overfitting
    warmup_epochs=3,     # Warmup period where learning rate increases gradually
    patience=10,         # Number of epochs with no improvement before stopping
    imgsz=640,           # Input image size (640x640 pixels)
    augment=True         # Apply augmentations
)

# Evaluation on the validation set after training
eval_results = model.val(data="config.yaml", iou=0.7,split='val', save=True)
print(eval_results)


print("------------ Data Training Terminated --------------")

