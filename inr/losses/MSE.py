import torch
from torch import nn

MSELoss = nn.MSELoss

class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred,target)) / torch.sqrt(torch.mean(torch.square(target)))
