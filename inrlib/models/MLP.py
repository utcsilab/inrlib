from lightning.pytorch.core.optimizer import LightningOptimizer
import numpy as np
from typing import Any, Tuple, Optional, Callable, Union, Mapping
import torch
from torch import nn
import lightning.pytorch as pl
from torch.optim.optimizer import Optimizer

from ..losses import ABCLoss 
from ..PosEncs import * 

from . import ABCModel 


class NeuralImplicitMLP(ABCModel):
    """
    Base 
    TinyNeRF: https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
    Fourier features are good notebook: https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
    """
    def __init__(self, 
          posenc: ABCPosEnc,
          loss_fn: nn.Module = nn.MSELoss(),
          linear_fn: nn.Module = nn.Linear,
          act_fn: nn.Module = nn.ReLU(),
          norm_fn: nn.Module = nn.Identity(),
          output_fn: nn.Module = nn.Identity(),
          init_fn: Optional[Callable]=None, 
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

        self.posenc = posenc
        self.loss_fn = loss_fn
        self.linear_fn = linear_fn
        self.act_fn = act_fn
        self.norm_fn = norm_fn
        self.output_fn = output_fn
        self.init_fn = init_fn
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_output = n_output
        self.learning_rate = lr
        self.optimizer = optimizer
        self.metrics = metrics
        self.monitor = monitor
        self.ignore_keys = ignore_keys
        
        self.iscustom = isinstance(self.loss_fn, ABCLoss)
                
        # set up layers
        self.layers = []
        self.layers += [linear_fn(posenc.d_output, n_features), act_fn, norm_fn]
        for i in range(n_layers-2):
                self.layers += [linear_fn(n_features, n_features), act_fn, norm_fn]
        self.layers += [linear_fn(n_features, n_output), output_fn]

        self.base_model = nn.Sequential(*self.layers)
        # self.save_hyperparameters()
        
        if init_fn:
            self.base_model.apply(init_fn) 
    
    def setup(self, stage: str):
        # add optimizable recon 
        # if self.iscustom and self.trainer.datamodule:
        #     dataset = self.trainer.datamodule.train_dataloader().dataset
        #     self.loss_fn.set_params(dataset.param_inits)
        
        super().setup(stage)
    
    def reconstruct(self, image_shape: Tuple[int, ...], **kwargs) -> Mapping[str, np.ndarray]:		
        assert len(self.outputs) > 0, "No validation outputs to reconstruct image from."

        idx = np.concatenate([data['idx'] for data in self.outputs], axis=0)
        pred = np.concatenate([data['pred'] for data in self.outputs], axis=0)
        target = np.concatenate([data['target'] for data in self.outputs], axis=0)
        
        pred_out = np.zeros_like(pred)
        target_out = np.zeros_like(target)
        pred_out[idx] = pred
        target_out[idx] = target
        pred_out = pred_out.reshape(*image_shape, *pred.shape[1:])
        target_out = target_out.reshape(*image_shape, *target.shape[1:])
        out = {'target': target_out, 'pred': pred_out, 'target-pred': target_out - pred_out}

        if self.iscustom:
            out = self.loss_fn.reconstruct_params(image_shape=image_shape, 
                                              outputs=self.outputs, 
                                              **out)
        return out
    
    def compute_metrics(self, 
                        pred: torch.Tensor, 
                        target: torch.Tensor, 
                        stage: str, 
                        **other):

        loss = self.loss_fn(pred, target, **other) if self.iscustom else self.loss_fn(pred, target)
        
        scores = {
        f"{stage}_loss": loss,
        f"{stage}_pred_mean": torch.mean(pred),
        f"{stage}_target_mean": torch.mean(target),
        f"{stage}_pred_std": torch.std(pred),
        f"{stage}_target_std": torch.std(target),
        f"{stage}_pred_min": torch.min(pred if not torch.is_complex(pred) else torch.abs(pred)),
        f"{stage}_target_min": torch.min(target if not torch.is_complex(target) else torch.abs(target)),
        f"{stage}_pred_max": torch.max(pred if not torch.is_complex(pred) else torch.abs(pred)),
        f"{stage}_target_max": torch.max(target if not torch.is_complex(target) else torch.abs(target)),
        }

        for metric in self.metrics:
            name = metric.__class__.__name__
            scores[f"{stage}_{name}"] = metric(pred, target)
        
        self.scores = scores if not stage == 'train' else self.scores 
        return scores

    def forward(self, x, **kwargs):
        xo = self.posenc(x)
        return self.base_model(xo)

    def training_step(self, batch, batch_idx=0, **kwargs):
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

    def validation_step(self, batch, batch_idx=0, **kwargs):
        if self.iscustom:
            inputs = self.loss_fn.prepare_input(**batch) 
            y_hat = self(**inputs)
            outputs = self.loss_fn.prepare_output(y_hat=y_hat, **inputs)
        else:
            y_hat = self(**batch)
            outputs = {'pred': y_hat, 'target': batch['y'], **batch}

        self.outputs += [{key: val.detach().cpu().numpy().copy() for key, val in outputs.items()}]
        scores = self.compute_metrics(**outputs, stage='val')
        self.log_dict(scores, sync_dist=True)

        loss = scores['val_loss']
        return loss
    
    def test_step(self, batch, batch_idx=0, **kwargs):
        if self.iscustom:
            inputs = self.loss_fn.prepare_input(**batch) 
            y_hat = self(**inputs)
            outputs = self.loss_fn.prepare_output(y_hat=y_hat, **inputs)
        else:
            y_hat = self(**batch)
            outputs = {'pred': y_hat, 'target': batch['y'], **batch}

        self.outputs += [{key: val.detach().cpu().numpy().copy() for key, val in outputs.items()}]
        scores = self.compute_metrics(**outputs, stage='test')
        self.log_dict(scores, sync_dist=True)
        
        loss = scores['test_loss']
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.iscustom:
            inputs = self.loss_fn.prepare_input(**batch) 
            y_hat = self(**inputs)
            outputs = self.loss_fn.prepare_output(y_hat=y_hat, **inputs)
        else:
            y_hat = self(**batch)
            outputs = {'pred': y_hat, 'target': batch['y'], **batch} 
            
        self.outputs += [{key: val.detach().cpu().numpy().copy()for key, val in outputs.items()}]
        self.compute_metrics(**outputs, stage='predict')

        return self.outputs
        
    def configure_optimizers(self):
        lr = self.learning_rate
        model_params = self.base_model.parameters()
        loss_params = self.loss_fn.parameters() if self.iscustom else []
        params = set(list(model_params) + list(loss_params))
        optimizer = self.optimizer(params=params, lr=lr)
        return optimizer

