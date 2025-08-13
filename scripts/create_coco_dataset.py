import argparse
import os
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from tqdm import tqdm

RESOLUTION = (512, 512)
MAX_PATCHES = 50
BLUR_SIGMA = (5, 40)
PATCH_FRAC = 0.1


def create_mask(img_size, patch_count, patch_frac):
    img_array = np.zeros(img_size)
    H, W = img_size
    patch_radius = int(max(H, W) * patch_frac) // 2
    
    for _ in range(patch_count):
        center_y = np.random.randint(patch_radius, H - patch_radius)
        center_x = np.random.randint(patch_radius, W - patch_radius)
        img_array[center_y - patch_radius:center_y + patch_radius, center_x - patch_radius:center_x + patch_radius] = 255

    return Image.fromarray(img_array.astype(np.uint8))


def create_gaussian_blurred_image(img, sigma):
    img_array = np.array(img)
    blurred_array = cv2.GaussianBlur(img_array, ksize=(0, 0), sigmaX=sigma)

    return Image.fromarray(blurred_array.astype(np.uint8))


def combine_control(target_img, mask, blurred_img):
    target_array = np.array(target_img)
    mask_array = np.array(mask)[:, :, np.newaxis] / 255.0
    blurred_array = np.array(blurred_img)

    assert target_array.shape == blurred_array.shape
    assert mask_array.shape == (*target_array.shape[:2], 1)

    combined_array = target_array * mask_array + blurred_array * (1 - mask_array)
    return Image.fromarray(combined_array.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/vmurugan/datasets')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train')
    parser.add_argument('--output_dir', type=str, default='outpaint')
    parser.add_argument('--combine_control', action='store_true')
    parser.add_argument('--save_mask', action='store_true')
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
    os.makedirs(os.path.join(output_dir, 'source'), exist_ok=True)
    if args.save_mask:
        os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)

    # Create resize transform
    resize_transform = transforms.Compose([
        transforms.Resize(RESOLUTION[0]),
        transforms.CenterCrop(RESOLUTION[0]),
    ])
    
    # List to store dataset entries
    dataset_entries = []
    
    print(f"Processing {len(image_ids)} images...")
    
    for image_id in tqdm(image_ids):
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
        
        # Create filenames
        filename = f"{image_id:08d}.jpg"
        
        # Process image
        target_img = resize_transform(img)

        # Generate random patch mask
        patch_count = np.random.randint(1, MAX_PATCHES + 1)
        mask = create_mask(RESOLUTION, patch_count, PATCH_FRAC)
        
        # Create blurred image
        sigma = np.random.randint(BLUR_SIGMA[0], BLUR_SIGMA[1] + 1)
        blurred_img = create_gaussian_blurred_image(target_img, sigma)

        # Combine control
        if args.combine_control:
            source_img = combine_control(target_img, mask, blurred_img)
        else:
            source_img = blurred_img

        # Save images
        target_img_path = os.path.join(output_dir, 'target', filename)
        source_img_path = os.path.join(output_dir, 'source', filename)

        target_img.save(target_img_path)
        source_img.save(source_img_path)

        if args.save_mask:
            mask_path = os.path.join(output_dir, 'mask', filename)
            mask.save(mask_path)

        # Create dataset entry
        entry = {
            "source": 'source/' + filename,
            "target": 'target/' + filename,
            "prompt": prompt
        }
        if args.save_mask:
            entry['mask'] = 'mask/' + filename
        dataset_entries.append(entry)
        
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
