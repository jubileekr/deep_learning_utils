import json
import os
from PIL import Image, ImageDraw, ImageChops

# Load JSON data from file
with open('instances_default_t.json') as f:
    data = json.load(f)

# Map category IDs to RGB colors
category_colors = {
    1: (255, 0, 0),   # Red
    2: (0, 255, 0),   # Green
    3: (0, 0, 255),   # Blue
    4: (255, 255, 0)  # Yellow
}

# Create blended image
for image in data['images']:
    image_id = image['id']
    image_pathrgb = os.path.join('imgs/train', image['file_name'])
    image_pathL = image_pathrgb
    #img = Image.open(image_pathrgb)
    maskrgb = Image.new('RGB', (384,320), color=(0, 0, 0))
    maskL = Image.new('L', (384,320))

    # Draw polygons for all categories in the image
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            color = category_colors.get(category_id)
            segmentation = annotation['segmentation'][0]
            polygon = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
            drawrgb = ImageDraw.Draw(maskrgb)
            drawL = ImageDraw.Draw(maskL)
            drawrgb.polygon(polygon, outline=color, fill=color)
            drawL.polygon(polygon, outline=category_id, fill=category_id)

    # Blend original image with maskd image
    #blended = Image.blend(img, maskrgb, alpha=0.1)

    # Save blended image with original filename in a different directory
    #image_namergb = os.path.basename(image_pathrgb)
    image_nameL = os.path.basename(image_pathL)

    base_name, ext = os.path.splitext(image_nameL)

# Set the file path and name for saving the image as a .png file
    save_pathL = os.path.join('pngs', base_name + '.png')
    #save_pathrgb = os.path.join('blended', image_namergb)
    #blended.save(save_pathrgb)
    maskL.save(save_pathL)
