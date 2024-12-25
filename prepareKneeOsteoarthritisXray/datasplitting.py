import os
import random
import shutil


def copy_random_images(source, destination, num_images, subfolder_map):
    for subfolder in subfolder_map:
        source_subfolder = os.path.join(source, subfolder)
        destination_subfolder = os.path.join(destination, subfolder_map[subfolder])

        image_files = [f for f in os.listdir(source_subfolder) if f.lower().endswith(".png")]

        selected_images = random.sample(image_files, num_images)
        print(f"copied {num_images} images from {source_subfolder} to {destination_subfolder}")

        for image in selected_images:
            source_path = os.path.join(source_subfolder, image)
            destination_path = os.path.join(destination_subfolder, image)
            shutil.copy(source_path, destination_path)

source_base = "./all_data"
destination_base = "./selected_data"

os.makedirs(os.path.join(destination_base, "train", "0"), exist_ok=True)
os.makedirs(os.path.join(destination_base, "train", "1"), exist_ok=True)
os.makedirs(os.path.join(destination_base, "test", "0"), exist_ok=True)
os.makedirs(os.path.join(destination_base, "test", "1"), exist_ok=True)

train_subfolder_map_0_1 = {"0": "0", "1": "0"}  # 0 and 1 go into train/0
train_subfolder_map_2_3_4 = {"2": "1", "3": "1", "4": "1"}  # 2, 3, 4 go into train/1
test_subfolder_map_0_1 = {"0": "0", "1": "0"}  # 0 and 1 go into test/0
test_subfolder_map_2_3_4 = {"2": "1", "3": "1", "4": "1"}  # 2, 3, 4 go into test/1

# train/0; 250 each from 0 and 1
copy_random_images(os.path.join(source_base, "train"), os.path.join(destination_base, "train"), 250, train_subfolder_map_0_1)

# train/1; 166 each from 2, 3, 4
copy_random_images(os.path.join(source_base, "train"), os.path.join(destination_base, "train"), 166, train_subfolder_map_2_3_4)

# test/0; 50 each from 0 and 1
copy_random_images(os.path.join(source_base, "test"), os.path.join(destination_base, "test"), 50, test_subfolder_map_0_1)

# test/1; 33 each from 2, 3, 4
copy_random_images(os.path.join(source_base, "test"), os.path.join(destination_base, "test"), 33, test_subfolder_map_2_3_4)
