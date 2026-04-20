import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
from torch import Tensor
from easydict import EasyDict
from models.graph_model import GraphModelWithVirtualNode, GraphModel, GraphModelWithMultipleVirtualNodes
from models.Sumformer import SumformerModel
from models.transformer import SetTransformerModel
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.data import Batch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from typing import List, Optional
import os


class StopAtValAccCallback(pl.Callback):
    """
    Callback for early stopping when validation accuracy reaches a target value.
    
    Args:
        target_acc (float): Accuracy threshold for stopping training early.
    """
    def __init__(self, target_acc=1.0):
        super().__init__()
        self.target_acc = target_acc

    def on_validation_epoch_end(self, trainer, _):
        """
        Checks validation accuracy at the end of each epoch, stopping training if the target is met.
        
        Args:
            trainer (Trainer): PyTorch Lightning trainer instance managing training.
        """
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_acc is not None and val_acc >= self.target_acc:
            trainer.should_stop = True
            print(f"Stopping training as `val_acc` reached {val_acc * 100:.2f}%")
        else:
            print(f"Current validation accuracy: {val_acc * 100:.2f}%")

        
                
class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a graph neural network model on various datasets.
    Supports distributed training on multiple GPUs using DDP (Distributed Data Parallel).
    
    Args:
        args (EasyDict): Configuration dictionary containing model parameters.
        model (GraphModel): The actual graph model instance to be trained/tested.
        task_id (int): Task index for multi-task training. Default is 0.
    """
    def __init__(self, args: EasyDict, model: GraphModel, task_id=0):
        super().__init__()
        self.task_id = task_id  
        self.lr = args.lr  
        self.lr_factor = args.lr_factor  
        self.optim_type = args.optim_type  
        self.weight_decay = args.wd 
        self.task_type = args.task_type  
        

        self.model = model
        
        self.save_hyperparameters(ignore=['model'])
    
    def on_train_start(self):
        """Called at the beginning of training. Useful for distributed setup."""
        if self.trainer.num_devices > 1:
            print(f"[GPU {self.global_rank}] Starting distributed training")
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        if self.trainer.num_devices > 1 and self.current_epoch == 0:
            print(f"[GPU {self.global_rank}] Epoch {self.current_epoch} started")
    
    def forward(self, X: Data) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            X (Data): Torch Geometric Data object containing node features and edge indices.
        
        Returns:
            Tensor: [num_nodes, out_dim] 
        """
        return self.model(X)

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        
        Returns:
            Tuple[List[Optimizer], Dict]: 
              - List containing the optimizer 
              - Dictionary with the LR scheduler config.
        """
        optimizer_cls = torch.optim.Adam if self.optim_type == 'Adam' else torch.optim.AdamW
        optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        lr_scheduler = ReduceLROnPlateau(
            optimizer, 
            factor=self.lr_factor,  
            mode='max',
            patience=2,        
        )
        
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "train_acc",  
        }
        return [optimizer], lr_scheduler_config


    def training_step(self, batch: Data, batch_idx):
        """
        Computes training loss and accuracy for a batch and logs them.
        """
        self.model.train()
        outputs = self(batch)  
        

        masked_indices = batch.train_mask.nonzero(as_tuple=True)[0]
        masked_outputs = outputs[masked_indices]
        

        labels = batch.y
        
        loss = F.cross_entropy(masked_outputs, labels)
        preds = torch.argmax(masked_outputs, dim=-1)
        acc = (preds == labels).float().mean()
        

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=labels.size(0))
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=labels.size(0))
        return loss


    def validation_step(self, batch: Data, batch_idx):
        self.model.eval()
        with torch.no_grad():
            outputs = self(batch)
            

            masked_indices = batch.val_mask.nonzero(as_tuple=True)[0]
            masked_outputs = outputs[masked_indices]
            

            labels = batch.y
            
            loss = F.cross_entropy(masked_outputs, labels)
            preds = torch.argmax(masked_outputs, dim=-1)
            acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=labels.size(0))
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=labels.size(0))
        return loss 


    def test_step(self, batch: Data, batch_idx):
        self.model.eval()
        with torch.no_grad():
            outputs = self(batch)
            

            masked_indices = batch.test_mask.nonzero(as_tuple=True)[0]
            masked_outputs = outputs[masked_indices]
            

            labels = batch.y
            
            loss = F.cross_entropy(masked_outputs, labels)
            preds = torch.argmax(masked_outputs, dim=-1)
            acc = (preds == labels).float().mean()
        
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=labels.size(0))
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=labels.size(0))
        return loss
        
        
class AccuracyPrintCallback(pl.Callback):
    """
    Simple callback to print accuracies every 50 epochs and summary at the end.
    """
    def __init__(self):
        super().__init__()
        self.accuracy_history = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_acc = trainer.callback_metrics.get('val_acc', 0.0)
        train_acc = trainer.callback_metrics.get('train_acc', 0.0)
        

        self.accuracy_history.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'val_acc': float(val_acc)
        })
        

        if (epoch + 1) % 50 == 0:
            print(f"\n[Epoch {epoch + 1}] Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%\n")
    
    def on_fit_end(self, trainer, pl_module):
        """Print summary at the end of training"""
        print("\n" + "="*60)
        print("TRAINING COMPLETE - ACCURACY SUMMARY")
        print("="*60)
        print(f"{'Epoch':<10} {'Train Acc':<15} {'Val Acc':<15}")
        print("-"*40)
        

        for record in self.accuracy_history:
            if record['epoch'] % 50 == 0:
                print(f"{record['epoch']:<10} {record['train_acc']*100:<15.2f} {record['val_acc']*100:<15.2f}")
        

        if self.accuracy_history and self.accuracy_history[-1]['epoch'] % 50 != 0:
            final = self.accuracy_history[-1]
            print(f"{final['epoch']:<10} {final['train_acc']*100:<15.2f} {final['val_acc']*100:<15.2f}")
        

        if self.accuracy_history:
            best_val = max(self.accuracy_history, key=lambda x: x['val_acc'])
            print(f"\nBest Val Accuracy: {best_val['val_acc']*100:.2f}% (Epoch {best_val['epoch']})")
        print("="*60 + "\n")