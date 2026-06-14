"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = nn.Dropout()
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
    





class ModelS4D(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 N=64, # State dim (poles per channel). 64 is a great default.
                 H=16, # Hidden channels. Try 16 or 32 for an amp model!
                 D=4): # Number of layers
        super().__init__()

        self.N = N
        self.H = H

        # Project 1D audio to H channels. 
        # bias=False is a GREAT choice here to prevent adding a DC offset!
        self.input_layer = nn.Linear(input_channels, self.H, bias=False)

        blocks = [S4D(d_model=self.H, d_state=N) for _ in range(D)]
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = nn.Linear(self.H, output_channels, bias=False)
        
        # Optional: A final hard-clipper for distortion
        self.act = nn.ReLU()

    def forward(self, x):
        # Initial input shape expected: (Batch, Length, 1)
        if x.ndim == 2:
            x = x.unsqueeze(-1) # Ensure shape is (B, L, 1)

        # 1. Project to Hidden Dimension
        # nn.Linear operates on the last dim. Shape becomes (B, L, H)
        x = self.input_layer(x)

        # 2. Transpose for S4D
        # S4D expects (Batch, Channels, Length)
        x = x.transpose(1, 2)

        # 3. Pass through S4D blocks
        for block in self.blocks:
            x, _ = block(x) # S4D returns (output, state)
            x = self.act(x)

        # 4. Transpose back for the output layer
        # Back to (Batch, Length, Channels)
        x = x.transpose(1, 2)

        # 5. Project back to 1 channel audio
        # Shape becomes (B, L, 1)
        y = self.output_layer(x)

        # 6. Apply final distortion clipping
        return y