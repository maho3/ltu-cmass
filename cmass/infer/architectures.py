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


class MLP(nn.Module):
    """
    Fully-connected network for embedding.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        act_fn: str = "ReLU",
    ):
        super().__init__()
        
        layers = []
        n_last = in_features
        for n_h in hidden_layers:
            layers.append(nn.Linear(n_last, n_h))
            layers.append(getattr(nn, act_fn)())
            n_last = n_h
        layers.append(nn.Linear(n_last, out_features))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiHeadEmbedding(nn.Module):
    """
    Multi-head embedding architecture.
    
    Takes a vector input, breaks it up into sub-vectors, embeds each
    individually with MLPs, and then concatenates the embeddings into a
    new context vector.
    """
    def __init__(
        self,
        start_idx: List[int],
        in_features: List[int],
        out_features: List[int],
        hidden_layers: List[List[int]],
        act_fn: str = "ReLU",
    ):
        super().__init__()
        
        self.start_idx = start_idx
        self.embedding_nets = nn.ModuleList()
        for i in range(len(start_idx)-1):
            self.embedding_nets.append(
                MLP(
                    in_features=in_features[i],
                    out_features=out_features[i],
                    hidden_layers=hidden_layers[i],
                    act_fn=act_fn,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the input tensor into sub-vectors
        sub_vectors = []
        for i in range(len(self.start_idx)-1):
            start = self.start_idx[i]
            end = self.start_idx[i+1]
            sub_vectors.append(x[..., start:end])
        
        # Embed each sub-vector
        embeddings = []
        for i, sub_vector in enumerate(sub_vectors):
            embeddings.append(self.embedding_nets[i](sub_vector))
            
        # Concatenate the embeddings
        return torch.cat(embeddings, dim=-1)


class FunnelNetwork(MLP):
    """
    Fully-connected network that linearly interpolates hidden features
    between input and output.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_depth: int,
        act_fn: str = "ReLU",
    ):
        hidden_layers = torch.linspace(
            in_features, out_features, hidden_depth+2
        ).to(torch.int)[1:-1].tolist()
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hidden_layers,
            act_fn=act_fn,
        )


class MultiHeadFunnel(nn.Module):
    """
    Multi-head embedding architecture with funnel networks.
    
    Takes a vector input, breaks it up into sub-vectors, embeds each
    individually with FunnelNetworks, and then concatenates the embeddings
    into a new context vector.
    """
    def __init__(
        self,
        start_idx: List[int],
        in_features: List[int],
        out_features: List[int],
        hidden_depth: List[int],
        act_fn: str = "ReLU",
    ):
        super().__init__()
        
        self.start_idx = start_idx
        self.embedding_nets = nn.ModuleList()
        for i in range(len(start_idx)-1):
            self.embedding_nets.append(
                FunnelNetwork(
                    in_features=in_features[i],
                    out_features=out_features[i],
                    hidden_depth=hidden_depth[i],
                    act_fn=act_fn,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the input tensor into sub-vectors
        sub_vectors = []
        for i in range(len(self.start_idx)-1):
            start = self.start_idx[i]
            end = self.start_idx[i+1]
            sub_vectors.append(x[..., start:end])
        
        # Embed each sub-vector
        embeddings = []
        for i, sub_vector in enumerate(sub_vectors):
            embeddings.append(self.embedding_nets[i](sub_vector))
            
        # Concatenate the embeddings
        return torch.cat(embeddings, dim=-1)
