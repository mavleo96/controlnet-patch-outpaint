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
from torchmetrics.image.lpip_similarity import LPIPS
from tqdm import tqdm
import json

def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    # Ensure tensor is in [0, 1] range
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy and PIL
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def compute_lpips_batch(reconstruction, generated, lpips_metric):
    """Compute LPIPS between reconstruction and generated images"""
    # Ensure tensors are in the correct format for LPIPS
    # LPIPS expects tensors in [0, 1] range and shape [B, C, H, W]
    if reconstruction.dim() == 3:
        reconstruction = reconstruction.unsqueeze(0)
    if generated.dim() == 3:
        generated = generated.unsqueeze(0)
    
    # Ensure tensors are in [0, 1] range
    if reconstruction.min() < 0:
        reconstruction = (reconstruction + 1.0) / 2.0
    if generated.min() < 0:
        generated = (generated + 1.0) / 2.0
    
    reconstruction = reconstruction.clamp(0, 1)
    generated = generated.clamp(0, 1)
    
    # Compute LPIPS
    lpips_score = lpips_metric(reconstruction, generated)
    return lpips_score.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/vmurugan/datasets/outpaint/val')
    parser.add_argument('--model_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='/data/vmurugan/controlnet-patch-outpaint/checkpoints/controlnet-patch-outpaint-continue-last-step=3999.ckpt')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./results/lpips_results.json')
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

    # Initialize LPIPS metric
    print("Initializing LPIPS metric...")
    lpips_metric = LPIPS().to(device)

    # Initialize LPIPS tracking
    lpips_results = {
        "control": [],
        "cfg_1": [],
        "cfg_4": []
    }
    
    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"Computing LPIPS over {total_samples} samples...")
    
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
            
            # Compute LPIPS for each sample in batch
            for i in range(args.batch_size):
                if batch_idx * args.batch_size + i >= total_samples:
                    break
                    
                lpips_results["control"].append(compute_lpips_batch(reconstruction[i], control[i], lpips_metric))
                lpips_results["cfg_1"].append(compute_lpips_batch(reconstruction[i], x_sample_cfg_1[i], lpips_metric))
                lpips_results["cfg_4"].append(compute_lpips_batch(reconstruction[i], x_sample_cfg_4[i], lpips_metric))
    
    # Calculate statistics
    print("\nComputing statistics...")
    stats = {}
    for method, lpips_values in lpips_results.items():
        if lpips_values:
            stats[method] = {
                "mean": np.mean(lpips_values),
                "std": np.std(lpips_values),
                "min": np.min(lpips_values),
                "max": np.max(lpips_values),
                "median": np.median(lpips_values),
                "count": len(lpips_values)
            }
    
    # Print results
    print("\nLPIPS Results:")
    print("=" * 60)
    for method, stat in stats.items():
        print(f"{method:12}: Mean={stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['count']})")
        print(f"{'':12}  Range: [{stat['min']:.4f}, {stat['max']:.4f}], Median: {stat['median']:.4f}")
        print()
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = {
        "lpips_values": lpips_results,
        "statistics": stats,
        "config": vars(args)
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
