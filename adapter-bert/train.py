import os
import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from dataset import GLUEDataModule
from model.model import GLUETransformer
from config import cfg
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adapter-Bert')
    parser.add_argument('--config', required=False, type=str, help='path to yaml config')
    args = parser.parse_args()
    if args.config:
        cfg.merge_from_file(args.config)

    seed_everything(cfg.RNG_SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = 'False'
    wandb_logger = WandbLogger(name=f'{cfg.MODEL_NAME}-{cfg.TRAINING_STRATEGY}',
                               project='adapter-bert')
    
    dm = GLUEDataModule(model_name_or_path=cfg.MODEL_NAME, 
                        task_name=cfg.TASK_NAME,
                        max_seq_length=cfg.MAX_SEQ_LENGTH,
                        train_batch_size=cfg.TRAIN_BATCH,
                        eval_batch_size=cfg.VAL_BATCH,
                        num_workers=cfg.NUM_WORKERS)

    dm.prepare_data()
    dm.setup("fit")

    warmup_steps = int(0.1 * len(dm.dataset['train'])) * cfg.EPOCHS
    
    model = GLUETransformer(
        model_name_or_path=cfg.MODEL_NAME,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
        strategy=cfg.TRAINING_STRATEGY,
        learning_rate=cfg.LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=cfg.WEIGHT_DECAY,
        train_batch_size=cfg.TRAIN_BATCH,
        eval_batch_size=cfg.VAL_BATCH,
        eval_splits=dm.eval_splits
    )
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    
    trainer = Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.NUM_GPUS if torch.cuda.is_available() else None,
        logger=wandb_logger,
        enable_checkpointing=False,
        deterministic=True,
        callbacks=callbacks
    )
    
    trainer.fit(model, datamodule=dm)
    
    wandb.config.update(
        {
        "warmup_steps": warmup_steps,
        "random_seed": cfg.RNG_SEED, 
        "max_sequence_length": cfg.MAX_SEQ_LENGTH,
        "num_gpus": cfg.NUM_GPUS,
        "num_workers": cfg.NUM_WORKERS
        })