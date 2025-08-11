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


def sampling_with_cfg_on_text(model, batch, N=1, ddim_steps=50, ddim_eta=0.0, unconditional_guidance_scale=9.0):
    use_ddim = ddim_steps is not None

    z, c = model.get_input(batch, model.first_stage_key, bs=N)
    c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
    N = min(z.shape[0], N)

    uc_cross = model.get_unconditional_conditioning(N)
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    samples_cfg, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                    batch_size=N, ddim=use_ddim,
                                    ddim_steps=ddim_steps, eta=ddim_eta,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=uc_full,
                                    )
    x_samples_cfg = model.decode_first_stage(samples_cfg)
    return x_samples_cfg

def sampling_without_text(model, batch, N=1, ddim_steps=50, ddim_eta=0.0):
    use_ddim = ddim_steps is not None

    z, c = model.get_input(batch, model.first_stage_key, bs=N)
    uc_cross = model.get_unconditional_conditioning(N)
    c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
    N = min(z.shape[0], N)

    samples, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [uc_cross]},
                                  batch_size=N, ddim=use_ddim,
                                  ddim_steps=ddim_steps, eta=ddim_eta,
                                  )
    x_samples = model.decode_first_stage(samples)
    
    return x_samples

def sample_with_text_only(model, batch, N=1, ddim_steps=50, ddim_eta=0.0):
    use_ddim = ddim_steps is not None
    z, c = model.get_input(batch, model.first_stage_key, bs=N)
    uc_cross = model.get_unconditional_conditioning(N)
    c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
    N = min(z.shape[0], N)
    
    ddim_sampler = DDIMSampler(model)
    samples, _ = ddim_sampler.sample(ddim_steps, N, (model.channels, *z.shape[2:]), {"c_crossattn": [c]},
        unconditional_conditioning={"c_crossattn": [uc_cross]},
        unconditional_guidance_scale=4.0, verbose=False)
    x_samples = model.decode_first_stage(samples)

    return x_samples


def create_grid_image(all_images, n_samples, output_path):
    # Column headers
    headers = ["Text", "Control", "Reconstruction", "Text only (CFG=4)", "No text", "CFG=1", "CFG=4"]
    
    # Create matplotlib subplots
    fig, axes = plt.subplots(n_samples, 7, figsize=(8 * 7, 8 * n_samples))
    
    # Add data rows with headers
    for row in range(n_samples):
        for col in range(7):
            img = all_images[col][row]
            img = (img + 1.0) / 2.0
            img = img.clamp(0, 1)
            img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            
            # Add header only to first row
            if row == 0:
                axes[row, col].set_title(headers[col], fontsize=60, pad=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/vmurugan/datasets/outpaint/val')
    parser.add_argument('--model_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='/data/vmurugan/controlnet-patch-outpaint/checkpoints/controlnet-patch-outpaint-continue-last-step=3999.ckpt')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./results/grid.png')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    device = args.device

    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'))

    model = model.to(device)
    model.cond_stage_model = model.cond_stage_model.to(device)
    model.cond_stage_model.device = device
    model.eval()

    dataset = OutPaintDataset(dataset_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    # Lists to store all images for grid creation
    all_text_conditioning = []
    all_control = []
    all_reconstruction = []
    all_x_sample_text_only = []
    all_x_sample = []
    all_x_sample_cfg_a = []
    all_x_sample_cfg_b = []

    DDIM_ETA = args.ddim_eta

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.n_samples:
            break
            
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        print(f"Processing batch {batch_idx + 1}/{args.n_samples}")
        
        # Generate all requested outputs
        with torch.no_grad():
            control = batch['hint'].permute(0, 3, 1, 2) * 2.0 - 1.0 # done here because create_grid_image expects [-1, 1]
            text_conditioning = log_txt_as_img((512, 512), batch['txt'], size=16)
            
            # Generate reconstruction (autoencoder reconstruction of source)
            z, _ = model.get_input(batch, model.first_stage_key, bs=args.batch_size)
            reconstruction = model.decode_first_stage(z)

            # Generate x_sample_text_only (without control)
            x_sample_text_only = sample_with_text_only(model, batch, args.batch_size, ddim_eta=DDIM_ETA)

            # Generate x_sample (unconditional sampling)
            x_sample = sampling_without_text(model, batch, args.batch_size, ddim_eta=DDIM_ETA)
            
            # Generate x_sample_cfg_1 (without classifier-free guidance, guidance_scale=1)
            x_sample_cfg_a = sampling_with_cfg_on_text(
                model, batch, args.batch_size,
                unconditional_guidance_scale=1.0,
                ddim_eta=DDIM_ETA
            )
            
            # Generate x_sample_cfg_9 (with classifier-free guidance, guidance_scale=9)
            x_sample_cfg_b = sampling_with_cfg_on_text(
                model, batch, args.batch_size, 
                unconditional_guidance_scale=4.0,
                ddim_eta=DDIM_ETA
            )
        
        # Store images for grid creation (take first image from batch)
        all_text_conditioning.append(text_conditioning[0] if text_conditioning.dim() > 3 else text_conditioning)
        all_control.append(control[0] if control.dim() > 3 else control)
        all_reconstruction.append(reconstruction[0] if reconstruction.dim() > 3 else reconstruction)
        all_x_sample_text_only.append(x_sample_text_only[0] if x_sample_text_only.dim() > 3 else x_sample_text_only)
        all_x_sample.append(x_sample[0] if x_sample.dim() > 3 else x_sample)
        all_x_sample_cfg_a.append(x_sample_cfg_a[0] if x_sample_cfg_a.dim() > 3 else x_sample_cfg_a)
        all_x_sample_cfg_b.append(x_sample_cfg_b[0] if x_sample_cfg_b.dim() > 3 else x_sample_cfg_b)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Combine all images in the specified order for grid
    all_images = [
        all_text_conditioning,
        all_control,
        all_reconstruction,
        all_x_sample_text_only,
        all_x_sample,
        all_x_sample_cfg_a,
        all_x_sample_cfg_b
    ]
    
    # Create and save grid image
    create_grid_image(all_images, args.n_samples, args.output_path)
    
    print(f"Completed! Grid image saved to {args.output_path}")


if __name__ == "__main__":
    main()

