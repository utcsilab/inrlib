from lightning.pytorch.core.optimizer import LightningOptimizer
import numpy as np
from typing import Any, Tuple, Optional, Callable, Union, Mapping
import torch
from torch import nn
import lightning.pytorch as pl
from torch.optim.optimizer import Optimizer

from inr.losses import ABCLoss 
from inr.PosEncs import * 

from . import ABCModel 


class NeuralImplicitMLP(ABCModel):
    """
    Base 
    TinyNeRF: https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
    Fourier features are good notebook: https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
    """
    def __init__(self, 
          posenc: nn.Module,
          loss_fn: nn.Module = nn.MSELoss(),
          act_fn: nn.Module = nn.ReLU(),
          output_fn: Optional[nn.Module] = None,
          n_features=256, 
          n_layers=4,
          n_output=2,
          lr=1e-3,
          optimizer: Callable=torch.optim.Adam,
          metrics=[],
          monitor=None,
          ignore_keys=[],
          **kwargs):
        super().__init__()

        self.n_features = n_features
        self.learning_rate = lr
        self.posenc = posenc
        self.loss_fn = loss_fn
        self.act_fn = act_fn
        self.output_fn = output_fn
        self.metrics = metrics
        self.monitor = monitor
        self.optimizer = optimizer
        
        self.iscustom = isinstance(self.loss_fn, ABCLoss)
                
        # set up layers
        self.layers = []
        self.layers += [nn.Linear(self.posenc.d_output, n_features), self.act_fn]
        for i in range(n_layers-2):
                self.layers += [nn.Linear(n_features, n_features), self.act_fn]
        self.layers += [nn.Linear(n_features, n_output)]
        if self.output_fn is not None:
            self.layers += [self.output_fn]

        self.base_model = nn.Sequential(*self.layers)
        # self.save_hyperparameters()
    
    def setup(self, stage: str):
        # add optimizable recon 
        if self.iscustom and self.trainer.datamodule:
            dataset = self.trainer.datamodule.train_dataloader().dataset
            self.loss_fn.set_params(dataset.input_shape)
        
        super().setup(stage)
    
    def reconstruct(self, image_shape: Tuple[int, ...], **kwargs) -> Mapping[str, np.ndarray]:		
        assert len(self.val_outputs) > 0, "No validation outputs to reconstruct image from."

        idx = np.concatenate([data['idx'] for data in self.val_outputs], axis=0)
        pred = np.concatenate([data['pred'] for data in self.val_outputs], axis=0)
        target = np.concatenate([data['target'] for data in self.val_outputs], axis=0)
        
        pred_out = np.zeros(pred.shape)
        target_out = np.zeros(target.shape)
        pred_out[idx] = pred
        target_out[idx] = target
        pred_out = pred_out.reshape(*image_shape, *pred.shape[1:])
        target_out = target_out.reshape(*image_shape, *target.shape[1:])
        out = {'target': target_out, 'pred': pred_out, 'target-pred': target_out - pred_out}

        if self.iscustom:
            out = self.loss_fn.reconstruct_params(image_shape=image_shape, 
                                              outputs=self.val_outputs, 
                                              **out)
        return out
    
    def compute_metrics(self, 
                        pred: torch.Tensor, 
                        target: torch.Tensor, 
                        stage: str, 
                        **kwargs):

        loss = self.loss_fn(pred, target)
        
        scores = {
        f"{stage}_loss": loss,
        f"{stage}_pred_mean": torch.mean(pred),
        f"{stage}_target_mean": torch.mean(target),
        f"{stage}_pred_std": torch.std(pred),
        f"{stage}_target_std": torch.std(target),
        f"{stage}_pred_min": torch.min(pred),
        f"{stage}_target_min": torch.min(target),
        f"{stage}_pred_max": torch.max(pred),
        f"{stage}_target_max": torch.max(target),
        }

        for metric in self.metrics:
            name = metric.__class__.__name__
            scores[f"{stage}_{name}"] = metric(pred, target)
        
        self.scores = scores if stage == 'val' else self.scores
        return scores

    def forward(self, x, **kwargs):
        xo = self.posenc(x)
        return self.base_model(xo)

    def training_step(self, batch, batch_idx, **kwargs):
        if self.iscustom:
            inputs = self.loss_fn.prepare_input(**batch) 
            y_hat = self(**inputs)
            outputs = self.loss_fn.prepare_output(y_hat=y_hat, **inputs)
        else:
            y_hat = self(**batch)
            outputs = {'pred': y_hat, 'target': batch['y'], **batch}

        scores = self.compute_metrics(**outputs, stage='train')
        self.log_dict(scores, sync_dist=True)

        loss = scores['train_loss']
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        if self.iscustom:
            inputs = self.loss_fn.prepare_input(**batch) 
            y_hat = self(**inputs)
            outputs = self.loss_fn.prepare_output(y_hat=y_hat, **inputs)
        else:
            y_hat = self(**batch)
            outputs = {'pred': y_hat, 'target': batch['y'], **batch}

        scores = self.compute_metrics(**outputs, stage='val')
        self.log_dict(scores, sync_dist=True)
        
        self.val_outputs += [{key: val.detach().cpu().numpy() for key, val in outputs.items()}]

        loss = scores['val_loss']
        return loss
        
    def configure_optimizers(self):
        lr = self.learning_rate
        model_params = self.base_model.parameters()
        loss_params = self.loss_fn.parameters() if self.iscustom else []
        params = list(model_params) + list(loss_params)
        optimizer = self.optimizer(params=params, lr=lr)
        return optimizer


## ADD YOUR CLASSES BELOW THAT INHERIT FROM ABOVE
    