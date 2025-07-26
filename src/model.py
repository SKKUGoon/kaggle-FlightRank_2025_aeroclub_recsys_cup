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
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h  # Update input dimension for next layer

        layers.append(nn.Linear(in_dim, 1))  # Output layer

        self.net: nn.Sequential = nn.Sequential(*layers)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        Input: x with shape (batch size, n features)
        Output: scores with shape (batch size, 
        """
        out: Tensor = self.net(x)
        return out.squeeze(-1) # (batch size,)


# Building on the study of
# https://arxiv.org/pdf/1912.05891
class SetRank(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        1. Setup
        Group of G items
        Each item has a feature vector of dimension d_m

        X (input feature matrix for the Group)

        Q: When acting as a query (what am I looking for)
        K: What do I offer
        V: What information do I carry

        Because it's multi-head - Q, K, V weights for each head

        Each item looks at every other item in the group
        Attention weight tells the model which items are most relevant when deciding final score
        The process is permutation invariant
        """
        super().__init__()
        
        # Initial item encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Transformer-style encoder layers for set-wise interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,  # Items can now attend to each other 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,  # [B, G, H] convention
            activation='gelu',
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Scorer
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [G, D] for a single group
        returns: [G] scores

        Output for each item depends on not just on its own features, but on the whole set
        """
        x = x.unsqueeze(0)  # Add a batch dimension: [1, G, D]
        h = self.encoder(x)  # Encode each item [1, G, D]
        h_ctx = self.self_attention(h)  # Contextualize with self-attention: [1, G, H]
        scores = self.scorer(h_ctx).squeeze(0).squeeze(-1)  # [G]
        return scores