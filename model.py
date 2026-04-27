import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba2 import Mamba2




class TanhApprox(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(1 + x**2)


class Model(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 N=8,
                 H=8,
                 D=4):
        super().__init__()

        self.N = N
        self.H = H

        self.input_layer = nn.Linear(input_channels, self.H, bias=False)

        blocks = [ModelBlock(N=self.N, H=self.H) for _ in range(D)]
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = nn.Linear(self.H, output_channels, bias=False)


    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None]

        x = self.input_layer(x)                                                 # Map input to hidden dimension H

        for block in self.blocks:                                               # Iterate over model blocks. Each block
            x = block(x)                                                        # consists of Mamba2 + NL w/skip.

        return self.output_layer(x)                                             # Map output back to original dimension


# ----------------------------------------------------------------------------- #

class ModelBlock(nn.Module):
    def __init__(self,
                 N,
                 H):
        super().__init__()

        self.ssm = Mamba2(d_model=H, 
                          d_state=N,
                          headdim=H//4,
                          ngroups=H//2, 
                          learnable_init_states=True)

        self.nonlinear_block = nn.Sequential(TanhApprox(),
                                             nn.Linear(H, H))

    def forward(self, x):

        y = self.ssm(x)
        y = self.nonlinear_block(y)

        return y + x

