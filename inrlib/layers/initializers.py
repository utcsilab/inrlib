import torch
import torch.nn as nn
from typing import List, Callable, Tuple, Mapping, Optional


class XavierNormal(nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, self.gain)
            nn.init.constant_(m.bias, 0)


class XavierUniform(nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, self.gain)
            nn.init.constant_(m.bias, 0)


class KaimingNormal(nn.Module):
    def __init__(self, a: float = 0., mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
        super().__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def forward(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, self.a, self.mode, self.nonlinearity)
            nn.init.constant_(m.bias, 0)


class KaimingUniform(nn.Module):
    def __init__(self, a: float = 0., mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
        super().__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def forward(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, self.a, self.mode, self.nonlinearity)
            nn.init.constant_(m.bias, 0)
