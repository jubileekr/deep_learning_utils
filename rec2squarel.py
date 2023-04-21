import cv2
import os

# Define input and output directories
input_dir = "/mnt/d/datasets/icu/pngs"
output_dir = "/mnt/d/datasets/icu/png2"


output_size = (224, 224)

# Loop over all bmp files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Read the image file in color mode
        image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Calculate the new size while preserving the aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = height / width
        new_height = int(output_size[0] * aspect_ratio)
        if new_height > output_size[1]:
            new_width = int(output_size[1] / aspect_ratio)
            size = (new_width, output_size[1])
        else:
            size = (output_size[0], new_height)

        # Resize the image
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        # Pad the image if necessary
        if size[1] != output_size[1]:
            #pad_width = output_size[0] - size[0]
            pad_height = output_size[1] - size[1]
            top_pad = pad_height //2
            bottom_pad = pad_height - top_pad
            left_pad = 0
            right_pad = 0
            pad = (top_pad, bottom_pad, left_pad, right_pad)
            resized_image = cv2.copyMakeBorder(resized_image, *pad, cv2.BORDER_CONSTANT, value=4)

        # Mirror the remaining empty space from the edge
#        mirrored_image = cv2.flip(resized_image, 1)

        # Save the image to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), resized_image)
