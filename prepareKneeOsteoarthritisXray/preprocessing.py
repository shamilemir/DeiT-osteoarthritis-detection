import cv2
import os
import numpy as np

def histogram_equalization(image):

    flat_image = image.flatten()
    histogram = np.zeros(256)
    
    for pixel in flat_image:
        histogram[pixel] += 1
    
    total_pixels = image.size
    probabilities = histogram / total_pixels

    cdf = np.cumsum(probabilities)

    intensity_levels = 256
    equalized = np.round((intensity_levels - 1) * cdf[image]).astype(np.uint8)

    return equalized

def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:

            # EQUALIZATION
            equalized_image = histogram_equalization(image)

            # RESIZING
            # resized_image = cv2.resize(equalized_image, (384, 384), interpolation=cv2.INTER_LINEAR)

            # SAVING
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, equalized_image) #or resized_image if resized

# ASSIGN INPUT AND OUTPUT FOLDERS
base_input_folder = "./selected_data"
base_output_folder = "./selected_processed_data"

#iterate
for folder in ["train", "test"]:
    for subfolder in ["0", "1"]:
        input_folder = os.path.join(base_input_folder, folder, subfolder)
        output_folder = os.path.join(base_output_folder, folder, subfolder)

        os.makedirs(output_folder, exist_ok=True)

        # PROCESS IMAGES
        process_images(input_folder, output_folder)
        print(f"Processed from {input_folder} into {output_folder}")
