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
```
# Local
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O /data/vmurugan/patch-out-painting/checkpoints/v1-5-pruned.ckpt
python tool_add_control.py /data/vmurugan/patch-out-painting/checkpoints/v1-5-pruned.ckpt /data/vmurugan/patch-out-painting/checkpoints/control_sd15_ini.ckpt

# Slurm
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt
python tool_add_control.py /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/v1-5-pruned.ckpt /home/vmurugan/projects/controlnet-patch-outpaint/checkpoints/control_sd15_ini.ckpt
```

## Train
```
# Local
python train.py \
    --root /home/vmurugan/ControlNet \
    --data_root /data/vmurugan \
    --dataset_dir datasets/outpaint \
    --checkpoints_dir patch-out-painting/checkpoints \
    --model_path models/cldm_v15.yaml \
    --resume_ckpt control_sd15_ini.ckpt \
    --max_steps 100 \
    --batch_size 1 \
    --accumulate_grad_batches 2 \
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
    --wandb_name single-control-only
```


