import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from . import ABCRegularizer


class L1Regularizer(ABCRegularizer):
    def __init__(self, dim: Optional[Tuple[int,...]] = None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.clone()
        for constraint in self.constraints:
            xi = constraint(xi)
        return self.weight * torch.sum(torch.linalg.norm(xi , ord=1, dim=self.dim))


class L2Regularizer(ABCRegularizer):
    def __init__(self, dim: Optional[Tuple[int,...]] = None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.clone()
        for constraint in self.constraints:
            xi = constraint(xi)
        return self.weight * torch.sum(torch.linalg.norm(xi , ord=2, dim=self.dim))
    
    
class TVRegularizer(ABCRegularizer):
    def __init__(self, dims: Optional[Tuple[int,...]] = None, ord: int=1, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims
        self.ord = ord
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.clone()
        for constraint in self.constraints:
            xi = constraint(xi)
        
        tv = 0. 
        for dim in self.dims: 
            tv += torch.sum(torch.abs(xi.diff(dim=dim, n=self.ord)))
            
        return self.weight * tv 
    
    
class L1ModelRegularizer(ABCRegularizer):
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        L1 Regularization loss for a model.

        Args:
            model (nn.Module): PyTorch model to regularize.
        Returns:
            torch.Tensor: L1 regularization loss.
        """
        l1_reg = torch.tensor(0.0)
        for param in model.parameters():
            l1_reg += torch.linalg.norm(param, ord=1)
        return self.weight * l1_reg


class L2ModelRegularizer(ABCRegularizer):
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        L1 Regularization loss for a model.

        Args:
            model (nn.Module): PyTorch model to regularize.
        Returns:
            torch.Tensor: L1 regularization loss.
        """
        l2_reg = torch.tensor(0.0)
        for param in model.parameters():
            l2_reg += torch.linalg.norm(param, ord=2)
        return self.weight * l2_reg

