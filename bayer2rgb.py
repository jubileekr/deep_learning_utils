import os
from PIL import Image
import numpy as np

# Specify the folder containing the BMP files
bmp_folder = "./val_o"

# Specify the folder to save the RGB files
rgb_folder = "./val"

# Loop over all BMP files in the folder
for filename in os.listdir(bmp_folder):
    if filename.endswith(".bmp"):
        # Load the Bayer pattern image
        bmp_path = os.path.join(bmp_folder, filename)
        img = Image.open(bmp_path)

        # Convert to numpy array
        arr = np.array(img)

        # Split the Bayer pattern into separate channels
        # Note that we assume the Bayer pattern is RGGB
        r = arr[::2, ::2]
        g1 = arr[1::2, ::2]
        g2 = arr[::2, 1::2]
        b = arr[1::2, 1::2]

        # Interpolate green values
        g = (g1 + g2) // 2

        # Combine into RGB image
        rgb_arr = np.dstack((r, g, b))

        # Convert back to PIL Image
        rgb_img = Image.fromarray(rgb_arr.astype('uint8'), 'RGB')

        # Save the RGB image to the same name in the RGB folder
        rgb_path = os.path.join(rgb_folder, filename)
        rgb_img.save(rgb_path)
