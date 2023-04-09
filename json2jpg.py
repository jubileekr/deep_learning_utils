import json
import os
from PIL import Image, ImageDraw

# Load JSON data from file
with open('annotations/instances_default.json') as f:
    data = json.load(f)

# Save indexed original images
for image in data['images']:
    image_path = os.path.join('images', image['file_name'])
    img = Image.open(image_path).convert('L')


# Create mask image
for annotation in data['annotations']:
    image_path = os.path.join('images', str(annotation['image_id']) + '.bmp')  # update file extension
    img = Image.new('L', (384, 320), color=0)  # set image dimensions
    
    # Create polygon mask
    segmentation = annotation['segmentation'][0]
    polygon = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
    
    # Draw mask
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, outline=1, fill=1)
    
    # Save mask image with original filename in a different directory
    image_name = next((image['file_name'] for image in data['images'] if image['id'] == annotation['image_id']), None)
    if image_name is None:
        continue
    save_path = os.path.join('labels', image_name)
    img = img.point(lambda x: x * 255)  # replace 1 with 255
    img.save(save_path)
    print(f"Created {save_path}")
