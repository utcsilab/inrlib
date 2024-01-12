import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Optional, List


class ABCRegularizer(ABC, nn.Module):
    def __init__(self, weight: float = 0.1, constraints: List[nn.Module]=[], **kwargs):
        super().__init__()
        self.weight = weight
        self.constraints = constraints # to apply to input before computing regularization loss


class L1Regularizer(ABCRegularizer):
    def __init__(self, dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for constraint in self.constraints:
            x = constraint(x)
        return self.weight * torch.linalg.norm(x, ord=1, dim=self.dim)


class L2Regularizer(ABCRegularizer):
    def __init__(self, dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for constraint in self.constraints:
            x = constraint(x)
        return self.weight * torch.linalg.norm(x, ord=2, dim=self.dim)
    

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
    

class TVRegularizer(ABCRegularizer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for constraint in self.constraints:
            x = constraint(x)
        return # TODO impl TV  
