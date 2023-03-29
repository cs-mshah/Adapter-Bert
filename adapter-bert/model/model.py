from datetime import datetime
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import evaluate
from model.bert import BertForSequenceClassification

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        strategy: str,
        learning_rate: float = 3e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        if self.hparams.strategy == 'full-finetuning':
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        elif self.hparams.strategy == 'adapter':
            self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        else:
            raise Exception("Unkown training strategy. Select from 'full-finetuning' or 'adapter'.")
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})

    def on_validation_epoch_end(self) -> None:
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True, logger=True, sync_dist=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True, logger=True, sync_dist=True)
            self.validation_step_outputs.clear()  # free memory
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        return loss
        
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer_grouped_parameters = []
        if self.hparams.strategy == 'full-finetuning':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        elif self.hparams.strategy == 'adapter':
            no_decay = ["adapter.proj_up.bias", "adapter.proj_down.bias", "LayerNorm"]
            cls_bias = ['cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias']
            cls_weight = ['cls.seq_relationship.weight', 'cls.predicions.transform.dense.weight', 'cls.predictions.decoder.weight']
            layers = ["adapter.proj_up.weight", "adapter.proj_down.weight"]
            layers.extend(cls_weight)
            no_decay.extend(cls_bias)
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]