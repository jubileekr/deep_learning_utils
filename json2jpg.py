import json
import os
from PIL import Image, ImageDraw, ImageChops

# Load JSON data from file
with open('annotations/instances_default.json') as f:
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
    #image_pathrgb = os.path.join('images', image['file_name'])
    image_pathL = os.path.join('images', image['file_name'])
    img = Image.open(image_pathrgb)
    maskrgb = Image.new('RGB', img.size, color=(0, 0, 0))
    maskL = Image.new('L', img.size)

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
    #save_pathrgb = os.path.join('blended', image_namergb)
    save_pathL = os.path.join('blended', image_nameL)
    #blended.save(save_pathrgb)
    maskL.save(save_pathL)
    
    print(f"Created {save_path}")

    save_path2 = os.path.join('labels', image_name)
    maskd.save(save_path2)
