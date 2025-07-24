from typing import List, Optional
import torch
from torch import nn, Tensor


class RankerNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]  # default hidden layers

        layers: List[nn.Module] = []
        in_dim = n_features  # Input dimension
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h  # Update input dimension for next layer

        layers.append(nn.Linear(in_dim, 1))  # Output layer

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        Input: x with shape (batch size, n features)
        Output: scores with shape (batch size, 
        """
        out: Tensor = self.net(x)
        return out.squeeze(-1) # (batch size,)

            