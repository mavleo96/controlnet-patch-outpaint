from share import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from outpaint.dataset import OutPaintDataset
from ldm.util import log_txt_as_img
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import os
from grid_inference import create_pil_image, sampling_with_cfg_on_text, sampling_without_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/vmurugan/datasets/outpaint/val')
    parser.add_argument('--model_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='/data/vmurugan/controlnet-patch-outpaint/checkpoints/controlnet-patch-outpaint-continue-last-step=3999.ckpt')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='/data/vmurugan/controlnet-patch-outpaint/results/controlnet_sd15')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    device = args.device
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'))

    model = model.to(device)
    model.cond_stage_model = model.cond_stage_model.to(device)
    model.cond_stage_model.device = device
    model.eval()

    dataset = OutPaintDataset(dataset_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    DDIM_ETA = args.ddim_eta

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.n_samples / args.batch_size:
            break
            
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        print(f"Processing batch {batch_idx + 1}/{args.n_samples / args.batch_size}")

        with torch.no_grad():
            x_sample_without_text = sampling_without_text(model, batch, args.batch_size, ddim_eta=DDIM_ETA)
            x_sample_with_text = sampling_with_cfg_on_text(model, batch, args.batch_size, unconditional_guidance_scale=1.0, ddim_eta=DDIM_ETA)

            os.makedirs(os.path.join(output_path, 'without_text'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'with_text'), exist_ok=True)

            for b in range(args.batch_size):
                img = create_pil_image(x_sample_without_text[b])
                img.save(os.path.join(output_path, 'without_text', batch['filename'][b]))
                img = create_pil_image(x_sample_with_text[b])
                img.save(os.path.join(output_path, 'with_text', batch['filename'][b]))

    print(f"Completed! Images saved to {output_path}")

if __name__ == "__main__":
    main()