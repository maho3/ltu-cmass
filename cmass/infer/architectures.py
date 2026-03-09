import torch
import torch.nn as nn
from typing import List

class CNN(nn.Module):
    def __init__(
        self,
        out_channels: List[int],
        kernel_size: int = 3,
        act_fn: str = "ReLU",
    ):
        super().__init__()
        
        layers = []
        n_last = 1
        for n_h in out_channels:
            layers.append(nn.Conv1d(n_last, n_h, kernel_size=kernel_size, padding='valid'))
            layers.append(getattr(nn, act_fn)())
            n_last = n_h
        
        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0) # (1, 1, Length)
        elif x.ndim == 2:
            x = x.unsqueeze(1) # (Batch, 1, Length)
        
        x = self.cnn(x)     # (Batch, Channels, New_Length)
        x = self.flatten(x) # (Batch, Channels * New_Length)
        return x