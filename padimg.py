import json
import os
from PIL import Image

input_dir = "/root/PIDNet/data/icu/test/images_o"
output_dir = "/root/PIDNet/data/icu/test/images"
json_dir = "/root/PIDNet/data/icu/test/annotations"

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    # Check if file is a .bmp image
    if filename.endswith(".bmp"):
        # Open the image using Pillow library
        img = Image.open(os.path.join(input_dir, filename))

        # Get the current dimensions of the image
        width, height = img.size

        # Calculate the amount of padding needed
        padding_width = 384 - width
        padding_height = 384 - height

        # Create a new image with the desired size and fill it with white color
        new_img = Image.new("L", (384, 384), 255)

        # Paste the original image onto the new image, offset by the padding amounts
        new_img.paste(img, (padding_width // 2, padding_height // 2))

        # Load the corresponding .json file
        json_file = os.path.join(json_dir, filename.replace(".bmp", ".json"))
        with open(json_file) as f:
            data = json.load(f)

        # Adjust the polygon mask coordinates to account for the padding
        for annotation in data["annotations"]:
            for point in annotation["polygon"]:
                point["x"] += padding_width // 2
                point["y"] += padding_height // 2

        # Save the padded image and updated .json file to the output directory with the same filename
        new_filename = os.path.join(output_dir, filename)
        new_img.save(new_filename)
        new_json_file = os.path.join(output_dir, filename.replace(".bmp", ".json"))
        with open(new_json_file, "w") as f:
            json.dump(data, f)
