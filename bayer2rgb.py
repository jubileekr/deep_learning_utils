import os
import cv2

input_folder = '/mnt/d/datasets/icu/imgs/val_o'
output_folder = '/mnt/d/datasets/icu/imgs/val'

# create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# loop over all BMP files in input folder
for filename in os.listdir(input_folder):
    if not filename.endswith('.bmp'):
        continue
    print(f"Processing {filename}...")
    bayer_img_path = os.path.join(input_folder, filename)

    # read bayer pattern image using OpenCV
    bayer_img = cv2.imread(bayer_img_path, cv2.IMREAD_GRAYSCALE)

    # interpolate missing color channels using OpenCV
    rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGRBG2RGB)

    # save the RGB image as a BMP file in the output folder
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, rgb_img)

print("Done!")
