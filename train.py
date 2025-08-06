from share import *
import os

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from outpaint.dataset import OutPaintDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb



def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--root', type=str, default='/home/vmurugan/ControlNet')
    parser.add_argument('--data_root', type=str, default='/data/vmurugan')
    parser.add_argument('--dataset_dir', type=str, default='datasets/outpaint')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--model_path', type=str, default='models/cldm_v15.yaml')
    parser.add_argument('--resume_ckpt', type=str, default='control_sd15_ini.ckpt')
    # Training
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--sd_locked', type=bool, default=True)
    parser.add_argument('--only_mid_control', type=bool, default=False)
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='controlnet-patch-outpaint')
    parser.add_argument('--wandb_name', type=str, default=None)
    args = parser.parse_args()
    
    # Paths
    dataset_path = os.path.join(args.data_root, args.dataset_dir, 'train')
    model_path = os.path.join(args.root, args.model_path)
    checkpoints_dir = os.path.join(args.data_root, args.checkpoints_dir)
    logs_dir = os.path.join(args.root, 'logs')
    resume_path = os.path.join(checkpoints_dir, args.resume_ckpt)

    # Training configs
    max_steps = args.max_steps
    batch_size = args.batch_size
    accumulate_grad_batches = args.accumulate_grad_batches
    learning_rate = args.learning_rate
    precision = args.precision if args.precision == 'bf16' else int(args.precision)
    sd_locked = args.sd_locked
    only_mid_control = args.only_mid_control
    
    # Wandb config
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    dataset = OutPaintDataset(dataset_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    image_logger = ImageLogger(batch_frequency=1000 * accumulate_grad_batches)
    model_checkpoint1 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='controlnet-patch-outpaint-last-{step}',
        every_n_train_steps=1000,
    )
    model_checkpoint2 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='controlnet-patch-outpaint-{step}',
        save_top_k=-1,
        every_n_train_steps=10000,
    )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        log_model=False,
        save_dir=logs_dir
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams({
        'model_path': model_path,
        'dataset_path': dataset_path,
        'resume_path': resume_path,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'sd_locked': sd_locked,
        'only_mid_control': only_mid_control,
        'img_logger_freq': 1000 * accumulate_grad_batches,
        'max_steps': max_steps,
        'accumulate_grad_batches': accumulate_grad_batches,
        'precision': precision
    })

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        max_steps=max_steps,
        callbacks=[image_logger, model_checkpoint1, model_checkpoint2],
        logger=wandb_logger
    )

    # Train!
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
