import torch
import torch.nn as nn


class ComplexLinear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features) + 1j * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features,) + 1j * torch.randn(out_features,))
        
    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        return torch.matmul(X, self.weight) + self.bias
        