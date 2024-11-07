import numpy as np
import torch
from torch import nn
from typing import Any, Literal, List, Callable, Tuple, Mapping, Optional
from abc import ABC, abstractmethod


class ABCLoss(ABC, nn.Module):
    def __init__(self, 
                 loss_type: Any, 
                 sample_inds: List[float] = [],
                 fncs: List[nn.Module] = [nn.MSELoss()], 
                 regularizers: List[nn.Module]=[], 
                 **kwargs
                 ):
        super().__init__()
        assert len(fncs) > 0, 'Specify at least one loss function'
        self.fncs = fncs
        self.loss_type = loss_type  # type of loss
        self.sample_inds = sample_inds  # indexing values if data contains multiple samples per image
        self.regularizers = regularizers

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = 0.
        for fnc in self.fncs:
            loss += fnc(pred,target)
        
        reg_val = 0.
        for reg in self.regularizers:
            reg_val += reg(pred, target, **kwargs)

        return loss + reg_val
    
    @abstractmethod
    def set_params(self) -> None:
        pass
    
    @abstractmethod
    def reconstruct_params(self) -> Mapping[str, np.ndarray]:
        pass

    @abstractmethod
    def prepare_input(self) -> Mapping[str, torch.Tensor]:
        pass

    @abstractmethod
    def prepare_output(self) -> Mapping[str, torch.Tensor]:
        pass


class ABCRegularizer(ABC, nn.Module):
    def __init__(self, weight: float = 0.1, 
                 constraints: List[nn.Module]=[],
                 update_weight: bool = False, 
                 **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight)) if update_weight else weight
        self.constraints = constraints # to apply to input before computing regularization loss
        