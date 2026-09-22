import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(approximate="tanh"),  # ?
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.block[0].weight)
        nn.init.zeros_(self.block[0].bias)
        nn.init.xavier_uniform_(self.block[2].weight)
        nn.init.zeros_(self.block[2].bias)

    def forward(self, x):
        return self.block(x)
