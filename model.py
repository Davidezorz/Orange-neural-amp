import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba2 import Mamba2
import math
from mambapy.mamba import Mamba, MambaConfig
# from mambapy.pscan import pscan
from pscan import pscan

import sys

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

        blocks = [ModelBlockLRU(N=self.N, H=self.H) for _ in range(D)]
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = nn.Linear(self.H, output_channels, bias=False)


    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None]

        x = self.input_layer(x)                                                 # Map input to hidden dimension H

        for block in self.blocks:                                               # Iterate over model blocks. Each block
            x = block(x)                                                        # consists of Mamba2 + NL w/skip.

        return self.output_layer(x)                                             # Map output back to original dimension
    

    def generate(self, x_input):
        """
        x_input: The raw audio input without the concatenated y_prev. 
                 Shape: (B, L, input_channels)
        """
        B, L, _ = x_input.shape
        device = x_input.device
        
        # Initialize empty states for every LRU block. Shape: (B, 1, 1, N)
        states = [torch.zeros(B, 1, 1, self.N, device=device) for _ in self.blocks]
        
        # Initialize the previous output as zeros for the very first step
        y_prev = torch.zeros(B, 1, 1, device=device)  # Assuming output_channels = 1
        
        outputs = []
        
        for t in range(L):
            if t%1000 == 0:
                sys.stdout.write(f"\rt: {t}/{L}")
                sys.stdout.flush()
                
            # 1. Grab the current input sample: (B, 1, input_channels)
            x_t = x_input[:, t:t+1, :]
            
            # 2. Concatenate the current input with the previous predicted output
            # This mimics your `y_input = torch.cat([y_input, y_output_prev], dim=-1)`
            combo_input = torch.cat([x_t, y_prev], dim=-1)
            
            # 3. Map to hidden dimension H
            hidden = self.input_layer(combo_input)
            
            # 4. Pass through all blocks sequentially, updating states
            for i, block in enumerate(self.blocks):
                hidden, states[i] = block.step(hidden, states[i])
                
            # 5. Map back to output dimension
            y_curr = self.output_layer(hidden)
            
            # 6. Store output and update y_prev for the next iteration
            outputs.append(y_curr)
            y_prev = y_curr
            
        # Concatenate all steps back into a single tensor (B, L, output_channels)
        return torch.cat(outputs, dim=1)


# ----------------------------------------------------------------------------- #

class ModelBlockMamba2(nn.Module):
    def __init__(self,
                 N,
                 H):
        super().__init__()

        self.ssm = Mamba2(d_model=H, 
                          d_state=N,
                          d_conv = 16,
                          headdim=H//4,
                          ngroups=1, 
                          learnable_init_states=True)

        self.nonlinear_block = nn.Sequential(TanhApprox(),
                                             nn.Linear(H, H))

    def forward(self, x):

        y = self.ssm(x)
        y = self.nonlinear_block(y)

        return y + x





# --------------------------------------------------------------------------- #


class ModelBlock3(nn.Module):
    def __init__(self,
                 N,
                 H):
        super().__init__()

        
        
        conf = MambaConfig(d_model=H, n_layers=1, d_state=N, d_conv=16)
        self.ssm = Mamba(conf)

        self.nonlinear_block = nn.Sequential(TanhApprox(),
                                             nn.Linear(H, H))

    def forward(self, x):

        y = self.ssm(x)
        y = self.nonlinear_block(y)

        return y + x



# --------------------------------------------------------------------------- #


class ModelBlockLRU(nn.Module):
    def __init__(  self, N, H):
        super().__init__()
        self.lru = LRUBlock(  N=N, H=H)
        """

        self.lru = MyLRUBlock(N=N, H=H)
        """

        self.nonlinear_block = nn.Sequential(TanhApprox(),
                                             nn.Linear(H, H))


    def forward(self, x):

        y = self.lru(x)
        y = self.nonlinear_block(y)

        return y + x


    def step(self, x_t, state_prev):
        """
        x_t: (B, 1, H)
        state_prev: (B, 1, 1, N)
        """
        # Step through the LRU
        y_t, state_t = self.lru.step(x_t, state_prev)
        
        # Apply the nonlinearity and residual connection
        out = self.nonlinear_block(y_t)
        
        return out + x_t, state_t


# --------------------------------------------------------------------------- #


class LRUBlock(nn.Module):
    def __init__(self, N, H):
        super().__init__()

        r_min = 0.8
        r_max = 1.0

        # Nu log
        u = torch.rand(1, 1, 1, N)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2)))
        self.nu_log._no_weight_decay = True

        # Gamma log
        A = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones(A.shape) - torch.abs(A) ** 2)))

        # B
        self.B = nn.Parameter(torch.randn(1, 1, N, H) / math.sqrt(2.0 * H))
        self.B._no_weight_decay = True

        # C
        self.C = nn.Parameter(torch.randn(1, 1, H, N) / math.sqrt(N))

        # D
        self.D = nn.Parameter(torch.randn(1, 1, H, 1)) 


    def scan(self, A, x):
        Ax = pscan(A, x)
        out = torch.zeros_like(Ax)
        out[:, 1:, ...] = Ax[:, :-1, ...]
        return out


    def forward(self, u):
        u = u.unsqueeze(-2)

        B, L, one, H = u.shape

        A = torch.exp(-torch.exp(self.nu_log))                                                      # 1 1 1 N
       
        # Expand A across batch/time dims
        A_expanded = A.expand(B, L, -1, -1)                                                         # B L 1 N
        
        # Ax + Bu
        Bu = torch.exp(self.gamma_log) * torch.matmul(self.B, u.transpose(-2, -1)).transpose(-2, -1) # B L 1 N
        x  = self.scan(A_expanded, Bu).transpose(-2, -1)                                             # B L N 1

        # Cx + Du
        y = torch.matmul(self.C, x)                                                                 # B L H 1
        y = y + self.D * u.transpose(-2, -1)                                                        # B L H 1

        return y.transpose(-2, -1).squeeze(-2)                                                                 # B L 1 H


    def step(self, u_t, x_prev):
        """
        Performs a single step of the LRU state space model.
        
        Args:
            u_t: Input at current time step -> Shape: (B, 1, H)
            x_prev: Hidden state from previous step -> Shape: (B, 1, 1, N)
            
        Returns:
            y_t: Output at current time step -> Shape: (B, 1, H)
            x_t: Updated hidden state -> Shape: (B, 1, 1, N)
        """
        u_t = u_t.unsqueeze(-2)                                                                     # B 1 1 H

        # 1. Fetch A
        A = torch.exp(-torch.exp(self.nu_log))                                                      # 1 1 1 N
        
        # 2. Compute B * u_t
        # matmul(B, u) yields (B, 1, N, 1). Transposing yields (B, 1, 1, N) to match x_prev
        Bu = torch.exp(self.gamma_log) * torch.matmul(self.B, u_t.transpose(-2, -1)).transpose(-2, -1) 

        # 3. Update hidden state: x_t = A * x_{t-1} + B * u_t
        x_t = A * x_prev + Bu                                                                       # B 1 1 N

        # 4. Compute output: y_t = C * x_t + D * u_t
        x_t_transposed = x_t.transpose(-2, -1)                                                      # B 1 N 1
        y_t = torch.matmul(self.C, x_t_transposed)                                                  # B 1 H 1
        y_t = y_t + self.D * u_t.transpose(-2, -1)                                                  # B 1 H 1

        # 5. Clean up dimensions for output
        y_out = y_t.transpose(-2, -1).squeeze(-2)                                                   # B 1 H
        
        return y_out, x_t



# --------------------------------------------------------------------------- #


class MyLRUBlock(nn.Module):
    def __init__(self, N, H, kernel_size=16):
        super().__init__()
        self.N, self.H = N, H


        self.conv1d = nn.Conv1d(in_channels=H+N, out_channels=H+N, 
                              kernel_size=kernel_size, bias=True, 
                              groups=H+N,
                              padding=kernel_size - 1)
        self.act = nn.SiLU()

        r_min = 0.8
        r_max = 1.0

        # Nu log
        u = torch.rand(1, 1, 1, N)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2)))
        self.nu_log._no_weight_decay = True

        # Gamma log
        A = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones(A.shape) - torch.abs(A) ** 2)))

        # B
        self.B = nn.Parameter(torch.randn(1, 1, N, H) / math.sqrt(2.0 * H))
        self.B._no_weight_decay = True

        # C
        self.C = nn.Parameter(torch.randn(1, 1, H, N) / math.sqrt(N))

        # D
        self.D = nn.Parameter(torch.randn(1, 1, H, 1)) 

        self.u_proj1 = nn.Linear(H, H+N)


    def scan(self, A, x):
        Ax = pscan(A, x)
        out = torch.zeros_like(Ax)
        out[:, 1:, ...] = Ax[:, :-1, ...]
        return out


    def forward(self, u):
        B, L, H = u.shape

        uB = self.u_proj1(u)

        # 1D Convolution
        uB = self.act(
            self.conv1d(uB.transpose(1, 2)).transpose(1, 2)
        )  # (B L H+N)
        uB = uB[:, :L, :]

        u, u_proj = torch.split(uB, [self.H, self.N], dim=-1)
        u_proj = u_proj[:, :, :, None]

        u = u.unsqueeze(-2)
        A = torch.exp(-torch.exp(self.nu_log))                                              # B L 1 N
       
        # Expand A across batch/time dims
        A_expanded = A.expand(B, L, -1, -1)                                                 # B L 1 N
        B_app = self.B*u_proj
        
        # Ax + Bu
        Bu = torch.exp(self.gamma_log) * (B_app @ u.transpose(-2, -1)).transpose(-2, -1)     # B L 1 N
        x  = self.scan(A_expanded, Bu).transpose(-2, -1)                                     # B L N 1

        # Cx + Du
        y = (self.C) @ x                                                                     # B L H 1
        y = y + self.D * u.transpose(-2, -1)                                                 # B L H 1

        return y.transpose(-2, -1).squeeze(-2)
