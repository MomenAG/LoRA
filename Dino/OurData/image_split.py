import os
import numpy as np
from PIL import Image


def find_split_point(image_path):
    with Image.open(image_path) as img:
        # Convert image to grayscale
        gray_img = img.convert('L')

        # Convert image to NumPy array
        data = np.array(gray_img)

        # Dimensions of the image
        height, width = data.shape

        # Define the regions to ignore
        ignore_top = int(0.25 * height)
        ignore_bottom = height - ignore_top
        ignore_left = int(0.2 * width)
        ignore_right = width - ignore_left

        # Crop the center region for analysis
        center_data = data[ignore_top:ignore_bottom, ignore_left:ignore_right]

        # Calculate brightness sums for each column in the center region
        col_brightness_sums = np.sum(center_data, axis=0)

        # Find the column index with the maximum brightness sum
        split_at = np.argmax(col_brightness_sums)

        # Adjust split_at to the full image coordinates
        split_at += ignore_left

        return split_at


def split_image(image_path, split_at, left_output_dir, right_output_dir):
    with Image.open(image_path) as img:
        width, height = img.size

        # Ensure split_at is within the image width
        split_at = min(split_at, width - 1)

        # Crop the image into two parts
        left_part = img.crop((0, 0, split_at, height))
        right_part = img.crop((split_at, 0, width, height))

        # Generate new filenames for left and right parts
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        left_image_path = os.path.join(left_output_dir, f"{base_filename}_left.jpg")
        right_image_path = os.path.join(right_output_dir, f"{base_filename}_right.jpg")

        # Save both parts
        left_part.save(left_image_path)
        right_part.save(right_image_path)

        print(f"Images have been split and saved as '{left_image_path}' and '{right_image_path}'.")


def process_images_in_directory(directory, left_output_dir, right_output_dir):
    # Create output directories if they don't exist
    os.makedirs(left_output_dir, exist_ok=True)
    os.makedirs(right_output_dir, exist_ok=True)

    # List all image files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(directory, filename)
            split_point = find_split_point(image_path)
            split_image(image_path, split_point, left_output_dir, right_output_dir)


# Usage
directory_path = 'class_1'  # Replace with your directory path containing the images
left_output_directory = 'class_1_left'  # Replace with your left output directory path
right_output_directory = 'class_1_right'  # Replace with your right output directory path
process_images_in_directory(directory_path, left_output_directory, right_output_directory)
