# controlnet-patch-outpaint

## Prepare Dataset
#### Download COCO Dataset
```
chmod +x scripts/download_coco_dataset.sh
./scripts/download_coco_dataset.sh /data/vmurugan/datasets
```

#### Create control and target images
```
python ./scripts/create_coco_dataset.py --root /data/vmurugan/datasets --split train --output_dir outpaint
python ./scripts/create_coco_dataset.py --root /data/vmurugan/datasets --split val --output_dir outpaint
```

## Train
#### Local
```
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
```

#### Slurm
```
python train.py \
    --root /home/vmurugan/projects/controlnet-patch-outpaint \
    --data_root /home/vmurugan/projects/controlnet-patch-outpaint \
    --dataset_dir datasets/outpaint \
    --checkpoints_dir checkpoints \
    --model_path models/cldm_v15.yaml \
    --resume_ckpt control_sd15_ini.ckpt \
    --max_steps 50000 \
    --batch_size 1 \
    --accumulate_grad_batches 2 \
    --wandb_name single-control-only
```


