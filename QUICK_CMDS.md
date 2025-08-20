# controlnet-patch-outpaint

## Setup
```
conda env create -f environment.yaml
conda activate control

wandb login
```

## Prepare Dataset
#### Download COCO Dataset
```
chmod +x scripts/download_coco_dataset.sh

# Local
./scripts/download_coco_dataset.sh /data/vmurugan/datasets

# Slurm
./scripts/download_coco_dataset.sh /home/vmurugan/projects/controlnet-patch-outpaint/datasets
```

#### Create control and target images
```
# Local
python ./scripts/create_coco_dataset.py --root /data/vmurugan/datasets --split train --output_dir outpaint
python ./scripts/create_coco_dataset.py --root /data/vmurugan/datasets --split val --output_dir outpaint

# Slurm
python ./scripts/create_coco_dataset.py --root /home/vmurugan/projects/controlnet-patch-outpaint/datasets --split train --output_dir outpaint
python ./scripts/create_coco_dataset.py --root /home/vmurugan/projects/controlnet-patch-outpaint/datasets --split val --output_dir outpaint
```

## Prepare Checkpoint (ControlNet)
#### Prepare Checkpoint (Stable Diffusion 1.5)
```
# Local
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O /data/vmurugan/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt
python tool_add_control.py \
    --input_path /data/vmurugan/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt \
    --output_path /data/vmurugan/controlnet-patch-outpaint/checkpoints/control_sd15_ini.ckpt \
    --model_path models/cldm_v15.yaml

# Slurm
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt
python tool_add_control.py \
    --input_path /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt \
    --output_path /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/control_sd15_ini.ckpt \
    --model_path models/cldm_v15.yaml
```

#### Prepare Checkpoint (Stable Diffusion 2.1)
```
# Local
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O /data/vmurugan/controlnet-patch-outpaint/checkpoints/v2-1_512-ema-pruned.ckpt
python tool_add_control.py \
    --input_path /data/vmurugan/controlnet-patch-outpaint/checkpoints/v2-1_512-ema-pruned.ckpt \
    --output_path /data/vmurugan/controlnet-patch-outpaint/checkpoints/control_sd21_ini.ckpt \
    --model_path models/cldm_v21.yaml
```

## Train
```
# Local
python train.py \
    --root /home/vmurugan/ControlNet \
    --data_root /data/vmurugan \
    --dataset_dir datasets/outpaint \
    --checkpoints_dir controlnet-patch-outpaint/checkpoints \
    --model_path models/cldm_v15.yaml \
    --resume_ckpt control_sd15_ini.ckpt \
    --max_steps 100 \
    --batch_size 1 \
    --accumulate_grad_batches 2 \
    --precision 16 \
    --wandb_name test-run

# Slurm
python train.py \
    --root /home/vmurugan/projects/controlnet-patch-outpaint \
    --data_root /home/vmurugan/projects/controlnet-patch-outpaint \
    --dataset_dir datasets/outpaint \
    --checkpoints_dir checkpoints \
    --model_path models/cldm_v15.yaml \
    --resume_ckpt control_sd15_ini.ckpt \
    --max_steps 50000 \
    --batch_size 16 \
    --accumulate_grad_batches 4 \
    --precision 16 \
    --wandb_name single-control-only
```

# Inference
```
python inference.py \
    --dataset_path '/data/vmurugan/datasets/outpaint/val' \
    --model_path 'models/cldm_v15.yaml' \
    --checkpoint_path '/data/vmurugan/controlnet-patch-outpaint/checkpoints/controlnet-patch-outpaint-continue-last-step=23999.ckpt' \
    --device 'cuda:0' \
    --n_samples 10 \
    --batch_size 1 \
    --output_path 'results/grid_10_bs64_e42999.png'
```