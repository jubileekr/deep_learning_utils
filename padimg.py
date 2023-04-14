import os
from PIL import Image

input_dir = "/root/PIDNet/data/tooth/train/images_o"
output_dir = "/root/PIDNet/data/tooth/train/images"

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    # Check if file is a .bmp image
    if filename.endswith(".bmp"):
        # Open the image using Pillow library
        img = Image.open(os.path.join(input_dir, filename))

        # Get the current dimensions of the image
        width, height = img.size

        # Calculate the amount of padding needed
        padding_height = 0  # 384 - height

        # Create a new image with the desired size and fill it with white color
        new_img = Image.new("L", (width, 384), 255)

        # Paste the original image onto the new image, offset by the padding amounts
        new_img.paste(img, (0, padding_height // 2))

        # Save the padded image to the output directory with the same filename
        new_filename = os.path.join(output_dir, filename)
        new_img.save(new_filename)
