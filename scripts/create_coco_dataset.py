import argparse
import os
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO

RESOLUTION = (512, 512)
MAX_PATCHES = 50
BLUR_SIGMA = (5, 40)
PATCH_FRAC = 0.1

# def create_patched_image(img, patch_frac, locations):
#     img_array = np.array(img)
#     H, W, _ = img_array.shape
    
#     patched_image = np.zeros_like(img_array)
#     patch_radius = int(max(H, W) * patch_frac) // 2

#     for center_y, center_x in locations:
#         y1 = max(center_y - patch_radius, 0)
#         y2 = min(center_y + patch_radius, H)
#         x1 = max(center_x - patch_radius, 0)
#         x2 = min(center_x + patch_radius, W)

#         patched_image[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]

#     return Image.fromarray(patched_image.astype(np.uint8))


def create_patched_image_with_blur(img, patch_frac, locations, sigma):
    img_array = np.array(img)
    H, W, _ = img_array.shape
    
    patch_radius = int(max(H, W) * patch_frac) // 2

    # Apply Gaussian blur to the entire image
    blurred_array = cv2.GaussianBlur(img_array, ksize=(0, 0), sigmaX=sigma)

    # Apply patches from original image
    for center_y, center_x in locations:
        y1 = max(center_y - patch_radius, 0)
        y2 = min(center_y + patch_radius, H)
        x1 = max(center_x - patch_radius, 0)
        x2 = min(center_x + patch_radius, W)
        blurred_array[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]

    return Image.fromarray(blurred_array.astype(np.uint8))

def resize_image(img, resolution):
    img_array = np.array(img)
    H, W, _ = img_array.shape

    # Resize
    scale = max(resolution[0] / H, resolution[1] / W)
    scale = np.ceil(scale * 100) / 100 # round up to ensure resolution is off by few pixels
    new_H, new_W = int(H * scale), int(W * scale)
    resized_array = cv2.resize(img_array, (new_W, new_H), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    # Crop to resolution (center crop)
    start_h, start_w = (new_H - resolution[0]) // 2, (new_W - resolution[1]) // 2
    cropped_array = resized_array[start_h:start_h+resolution[0], start_w:start_w+resolution[1]]
    assert cropped_array.shape[:2] == resolution

    return Image.fromarray(cropped_array.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/vmurugan/datasets')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train')
    parser.add_argument('--output_dir', type=str, default='outpaint')
    args = parser.parse_args()

    # Initialize COCO API for captions
    ann_file = os.path.join(args.root, 'coco', 'annotations', f'captions_{args.split}2017.json')
    coco = COCO(ann_file)
    
    # Get all image IDs
    image_ids = coco.getImgIds()
    
    # Create output directory structure
    output_dir = os.path.join(args.root, args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'patched'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'source'), exist_ok=True)
    
    # List to store dataset entries
    dataset_entries = []
    
    print(f"Processing {len(image_ids)} images...")
    
    for i, image_id in enumerate(image_ids):

        # Load image info
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        # Load image
        image_path = os.path.join(args.root, 'coco', f'{args.split}2017', file_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue
            
        img = Image.open(image_path)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get captions for this image
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in annotations]
        
        # Use the first caption as prompt (or combine them)
        prompt = captions[0] if captions else "A photograph"
        
        # Generate random patch locations
        patch_count = np.random.randint(1, MAX_PATCHES + 1)
        locations = (np.random.rand(patch_count, 2) * np.array(img.size)).astype(int)
        
        # Create filenames
        filename = f"{image_id:08d}.jpg"
        
        # Process image
        target_img = resize_image(img, RESOLUTION)
        # patched_img = create_patched_image(img, PATCH_FRAC, locations)
        sigma = np.random.randint(BLUR_SIGMA[0], BLUR_SIGMA[1] + 1)
        blurred_img = create_patched_image_with_blur(target_img, PATCH_FRAC, locations, sigma)
        
        # Save images
        target_img_path = os.path.join(output_dir, 'target', filename)
        # patched_img_path = os.path.join(output_dir, 'patched', filename)
        blurred_img_path = os.path.join(output_dir, 'source', filename)
        
        target_img.save(target_img_path)
        # patched_img.save(patched_img_path)
        blurred_img.save(blurred_img_path)
        
        # Create dataset entry
        entry = {
            "source": 'source/' + filename,
            "target": 'target/' + filename,
            "prompt": prompt
        }
        dataset_entries.append(entry)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_ids)} images")
    
    # Save dataset JSON file
    json_path = os.path.join(output_dir, 'prompt.json')
    with open(json_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Completed! Dataset saved to {output_dir}")
    print(f"Total entries: {len(dataset_entries)}")
    print(f"JSON file: {json_path}")


if __name__ == '__main__':
    main()
