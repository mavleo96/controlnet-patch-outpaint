from share import *
from inference import sampling_with_cfg_on_text

import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from outpaint.dataset import OutPaintDataset
from ldm.util import log_txt_as_img
from cldm.model import create_model, load_state_dict
import os
from torchmetrics.functional import psnr
from tqdm import tqdm
import json

def tensor_to_pil(tensor):
    """Convert tensor to PIL image for PSNR calculation"""
    # Ensure tensor is in [0, 1] range
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy and PIL
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def compute_psnr_batch(reconstruction, generated):
    """Compute PSNR between reconstruction and generated images"""
    # Convert tensors to PIL images
    recon_pil = tensor_to_pil(reconstruction)
    gen_pil = tensor_to_pil(generated)
    
    # Convert PIL to tensors for PSNR calculation
    recon_array = np.array(recon_pil).astype(np.float32) / 255.0
    gen_array = np.array(gen_pil).astype(np.float32) / 255.0
    
    recon_tensor = torch.from_numpy(recon_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    gen_tensor = torch.from_numpy(gen_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    return psnr(gen_tensor, recon_tensor, data_range=1.0).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/vmurugan/datasets/outpaint/val')
    parser.add_argument('--model_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='/data/vmurugan/controlnet-patch-outpaint/checkpoints/controlnet-patch-outpaint-continue-last-step=3999.ckpt')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./results/psnr_results.json')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process (None for all)')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    device = args.device

    # Load model
    print("Loading model...")
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'))
    model = model.to(device)
    model.cond_stage_model = model.cond_stage_model.to(device)
    model.cond_stage_model.device = device
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = OutPaintDataset(dataset_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    # Initialize PSNR tracking
    psnr_results = {
        "control": [],
        "cfg_1": [],
        "cfg_4": []
    }
    
    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"Computing PSNR over {total_samples} samples...")
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, total=total_samples//args.batch_size)):
        if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
            break
            
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            # Generate reconstruction (ground truth)
            z, _ = model.get_input(batch, model.first_stage_key, bs=args.batch_size)
            reconstruction = model.decode_first_stage(z)
            
            # Get control signal (hint)
            control = batch['hint'].permute(0, 3, 1, 2) * 2.0 - 1.0
            
            # Generate different outputs
            x_sample_cfg_1 = sampling_with_cfg_on_text(model, batch, args.batch_size, unconditional_guidance_scale=1.0, ddim_eta=args.ddim_eta)
            x_sample_cfg_4 = sampling_with_cfg_on_text(model, batch, args.batch_size, unconditional_guidance_scale=4.0, ddim_eta=args.ddim_eta)
            
            # Compute PSNR for each sample in batch
            for i in range(args.batch_size):
                if batch_idx * args.batch_size + i >= total_samples:
                    break
                    
                psnr_results["control"].append(compute_psnr_batch(reconstruction[i], control[i]))
                psnr_results["cfg_1"].append(compute_psnr_batch(reconstruction[i], x_sample_cfg_1[i]))
                psnr_results["cfg_4"].append(compute_psnr_batch(reconstruction[i], x_sample_cfg_4[i]))
    
    # Calculate statistics
    print("\nComputing statistics...")
    stats = {}
    for method, psnr_values in psnr_results.items():
        if psnr_values:
            stats[method] = {
                "mean": np.mean(psnr_values),
                "std": np.std(psnr_values),
                "min": np.min(psnr_values),
                "max": np.max(psnr_values),
                "median": np.median(psnr_values),
                "count": len(psnr_values)
            }
    
    # Print results
    print("\nPSNR Results:")
    print("=" * 60)
    for method, stat in stats.items():
        print(f"{method:12}: Mean={stat['mean']:.3f} ± {stat['std']:.3f} dB (n={stat['count']})")
        print(f"{'':12}  Range: [{stat['min']:.3f}, {stat['max']:.3f}] dB, Median: {stat['median']:.3f} dB")
        print()
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = {
        "psnr_values": psnr_results,
        "statistics": stats,
        "config": vars(args)
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
