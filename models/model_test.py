"""
Code by https://github.com/Davidezorz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pscan import pscan
import math



class TanhApprox(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(1 + x**2)





class ModelTest(nn.Module):

    def __init__(self, 
                 S:             int, 
                 K:             int, 
                 sampling_rate: float):
        """
        Args:
            - S             (int):    Number of cascaded processing blocks.
            - K             (int):    Number of biquad filters in each block.
            - sampling_rate (float):  Audio sampling rate in Hz.
        """
        super().__init__()
        self.K = K
        self.S = S

        blocks = [SelectiveBiquadsBlock(K=K, sampling_rate=sampling_rate) 
                  for _ in range(S)]
        self.blocks = nn.ModuleList(blocks)
        self.gains = nn.Parameter(torch.rand(S)+0.5)

        self.tanh = nn.Tanh() #TanhApprox()
        

    def forward(self, y):
        if y.ndim == 2:
            y = y[:, :, None]

        for i, block in enumerate(self.blocks):                                 # Iterate over model blocks. Each block
            # print(f"i: {i}")
            y = self.gains[i]*block(y)                                          # consists of filter + NL
            if i != len(self.blocks)-1:
                y = self.tanh(y)
        # print("end")
        return y



# -----------------------------------------------------------------------------

class BiquadsBlock(nn.Module):

    def __init__(self, K, sampling_rate):
        super().__init__()
        self.K = K
        self.sampling_rate = sampling_rate

        # parameters
        f_raw    = torch.randn(K)
        f_start  = 50 + f_raw[0]*20 
        val = max(-0.9, min(f_start*4*K/sampling_rate - 1, 0.9))
        f_raw[0] = math.asin(val)   # the first frequency should not be too small
        
        self.db_gain = nn.Parameter(7*(torch.rand(K)-0.5))
        self.f_raw   = nn.Parameter(f_raw)
        self.Q_raw   = nn.Parameter(torch.rand(K))


        self.db_gain._no_weight_decay = True
        self.f_raw._no_weight_decay   = True
        self.Q_raw._no_weight_decay   = True

        # other values
        Q_max = torch.ones_like(self.Q_raw)*3
        Q_max[0] = Q_max[-1] = 1
        self.register_buffer('Q_max', Q_max)

        self.b_0 = None
        self.b_1 = None
        self.b_2 = None

        self.a_1 = None
        self.a_2 = None
    

    def forward_pscan(self, x):        
        B, L, C = x.shape
        y = x.clone()
        
        # 0. CAST TO COMPLEX (float32)
        a1 = self.a_1.squeeze(-1).to(torch.cfloat)
        a2 = self.a_2.squeeze(-1).to(torch.cfloat)
        b0 = self.b_0.squeeze(-1).to(torch.cfloat)
        b1 = self.b_1.squeeze(-1).to(torch.cfloat)
        b2 = self.b_2.squeeze(-1).to(torch.cfloat)

        # 1. CALCULATE BOTH EIGENVALUES (The Poles)
        discriminant = a1**2 - 4*a2
        
        # We only need a microscopic epsilon now because float64 has 15 decimal places of precision
        sqrt_D = torch.where(torch.abs(discriminant) < 1e-12, 
                             torch.tensor(1e-12, dtype=torch.cfloat, device=x.device), 
                             torch.sqrt(discriminant))
        
        lam1 = (-a1 + sqrt_D) / 2.0
        lam2 = (-a1 - sqrt_D) / 2.0

        # 2. CALCULATE BOTH RESIDUES
        p1 = b1 - b0 * a1
        p2 = b2 - b0 * a2
        R1 = (p1 * lam1 + p2) / sqrt_D
        R2 = (p1 * lam2 + p2) / (-sqrt_D)

        # 3. PACK
        # Stack them so we can process both poles in parallel inside the scan!
        lam = torch.stack([lam1, lam2], dim=1).to(torch.cfloat) # Shape: (K, 2)
        R   = torch.stack([R1, R2], dim=1).to(torch.cfloat)     # Shape: (K, 2)
        b0  = b0.to(torch.cfloat)

        # 4. CASCADE THE FILTERS VIA PARALLEL SCAN
        for k in range(self.K):
            # A_scan: (B, L, C, 2) -> Holds both poles
            A_scan = lam[k].view(1, 1, 1, 2).expand(B, L, C, 2)
            
            # X_scan: (B, L, C, 2) -> Copy the input audio for both poles
            X_scan = y.unsqueeze(-1).expand(B, L, C, 2).to(torch.cfloat)
            
            # Compute BOTH poles simultaneously! Output: (B, L, C, 2)
            h = pscan(A_scan, X_scan) 
            
            # Shift the sequence to the right by 1 step
            h_delayed = torch.cat([
                torch.zeros(B, 1, C, 2, dtype=torch.cfloat, device=x.device), 
                h[:, :-1, :, :]
            ], dim=1)

            # 5. RECONSTRUCT THE AUDIO
            # Multiply each state by its residue, sum the 2 poles together, and take the real part
            y_rem = (R[k].view(1, 1, 1, 2) * h_delayed).sum(dim=-1)

            y = b0[k].real * y + y_rem.real

        return y
    

    def compute_coefficients(self):
        """https://arxiv.org/pdf/2103.08709
           https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html"""
    
        # compute f
        f_squished = (torch.sin(self.f_raw) + 1)*0.25 
        f_cumsum = torch.cumsum(f_squished, dim=0)                              # 3. force sequential ordering (f0 <= f1 <= f2...)
        f = (self.sampling_rate / self.K) * f_cumsum

        # compute Q
        Q = self.Q_max*F.sigmoid(self.Q_raw) + 1e-6

        # get A, w_0, alpha 
        A = 10**(self.db_gain/40)
        w_0 = 2*torch.pi*f/self.sampling_rate
        alpha = 0.5*torch.sin(w_0)/Q

        # compute coefficients
        a_0 =  1 + alpha/A

        self.b_0 = (1 + alpha*A)      / a_0
        self.b_1 = -2*torch.cos(w_0)  / a_0
        self.b_2 = (1 - alpha*A)      / a_0

        self.a_1 = -2*torch.cos(w_0)  / a_0
        self.a_2 = (1 - alpha/A)      / a_0


    def forward(self, x: torch.Tensor):
        self.compute_coefficients()
  
        if x.ndim == 2:                                                         # x is B L C
            x = x[:, :, None]

        return self.forward_pscan(x)
    











# -----------------------------------------------------------------------------


class SelectiveBiquadsBlock(nn.Module):
    def __init__(self, K, sampling_rate, in_channels=1):
        super().__init__()
        self.K = K
        self.sampling_rate = sampling_rate

        # --- STATIC BASE PARAMETERS ---
        f_raw    = torch.randn(K)
        f_start  = 50 + f_raw[0]*20
        val = max(-0.98, min(f_start*4*K/sampling_rate - 1, 0.98))
        f_raw[0] = math.asin(val)
        

        f_start  = 5
        val = f_start*K/sampling_rate
        self.f_safe = val


        self.db_gain = nn.Parameter(7*(torch.rand(K)-0.5))
        self.f_raw   = nn.Parameter(f_raw)
        self.Q_raw   = nn.Parameter(torch.rand(K))

        self.f_raw._no_weight_decay   = True
        self.db_gain._no_weight_decay = True
        self.Q_raw._no_weight_decay   = True

        Q_max = torch.ones_like(self.Q_raw)*3
        Q_max[0] = Q_max[-1] = 1
        self.register_buffer('Q_max', Q_max)

        # --- SELECTIVITY MECHANISM (NEW) ---
        # 1D Causal Convolution to look at past 16 samples
        self.kernel_size = 16
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=2 * K,  # We need delta_f, delta_Q, delta_gain for each filter!
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1, # Causal padding ensures we don't look into the future
            groups=1 
        )
        
        # CRITICAL: Initialize to zero so the model starts perfectly LTI and stable!
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)


    def compute_coefficients(self, x):
        B, L, C = x.shape
        
        # 1. COMPUTE TIME-VARYING DELTAS
        # Conv1d expects (Batch, Channels, Length)
        x_c = x.transpose(1, 2) 
        delta = F.tanh(self.conv(x_c))
        
        # Crop off the extra causal padding to perfectly match length L
        delta = delta[..., :L] 
        
        # Split the 3*K channels into our three parameter deltas. Each shape: (B, K, L)
        dQ, dgain = torch.chunk(delta, 2, dim=1) 
        
        # 2. ADD TO BASE PARAMETERS
        # Expand static params (K) to (1, K, 1) so they broadcast with the deltas

        f_raw_t   = self.f_raw.view(1, self.K, 1)
        Q_raw_t   = self.Q_raw.view(1, self.K, 1) + dQ
        db_gain_t = self.db_gain.view(1, self.K, 1) + dgain


        first_one = torch.zeros_like(f_raw_t)
        first_one[:, 0, :] = 1 
        choose = first_one == 1

        # 3. COMPUTE TIME-VARYING COEFFICIENTS
        f_squished = (torch.sin(f_raw_t) + 1)*0.25 
        f_squished = torch.where(choose, self.f_safe + f_squished, f_squished)
        # Note: We must cumsum across the K dimension (dim=1) now, not 0!
        f_cumsum = torch.cumsum(f_squished, dim=1)  
        f = (self.sampling_rate / self.K) * f_cumsum

        Q = self.Q_max.view(1, self.K, 1) * F.sigmoid(Q_raw_t) + 1e-8
        A = 10**(db_gain_t / 40)
        
        w_0 = 2 * torch.pi * f / self.sampling_rate
        alpha = 0.5 * torch.sin(w_0) / Q

        a_0 = 1 + alpha / A

        # The coefficients are now mathematically time-varying! Shape: (B, K, L)
        # We cast them straight to double precision here to keep the pscan clean
        self.b_0 = ((1 + alpha * A) / a_0).to(torch.cfloat)
        self.b_1 = (-2 * torch.cos(w_0) / a_0).to(torch.cfloat)
        self.b_2 = ((1 - alpha * A) / a_0).to(torch.cfloat)
        self.a_1 = (-2 * torch.cos(w_0) / a_0).to(torch.cfloat)
        self.a_2 = ((1 - alpha / A) / a_0).to(torch.cfloat)

        # Stability condition
        # self.a_2 = torch.clamp(self.a_2, min=-0.99, max=0.99)
        # self.a_1 = torch.clamp(self.a_1, min=(-0.99 - self.a_2), max=(self.a_2 + 0.99))


    def forward(self, x):
        # Pass x so the conv layer can analyze it!
        self.compute_coefficients(x) 
        
        B, L, C = x.shape
        y = x.clone()

        # The rest of the math remains IDENTICAL. PyTorch handles the (B, K, L) 
        # shapes automatically via element-wise operations!
        discriminant = self.a_1**2 - 4*self.a_2
        sqrt_D = torch.where(torch.abs(discriminant) < 1e-12, 
                             torch.tensor(1e-12, dtype=torch.cfloat, device=x.device), 
                             torch.sqrt(discriminant))
        
        lam1 = (-self.a_1 + sqrt_D) / 2.0
        lam2 = (-self.a_1 - sqrt_D) / 2.0

        p1 = self.b_1 - self.b_0 * self.a_1
        p2 = self.b_2 - self.b_0 * self.a_2
        R1 = (p1 * lam1 + p2) / sqrt_D
        R2 = (p1 * lam2 + p2) / (-sqrt_D)

        # Pack to (B, K, L, 2)
        lam = torch.stack([lam1, lam2], dim=-1).to(torch.cfloat) 
        R   = torch.stack([R1, R2], dim=-1).to(torch.cfloat)     
        b0  = self.b_0.to(torch.cfloat)

        for k in range(self.K):
            # Extract the L sequence for filter k. Shape becomes (B, L, 2)
            lam_k = lam[:, k, :, :]
            R_k   = R[:, k, :, :]
            b0_k  = b0[:, k, :]
            
            # Expand to include the Channel dimension for pscan: (B, L, C, 2)
            A_scan = lam_k.unsqueeze(2).expand(B, L, C, 2)
            X_scan = y.unsqueeze(-1).expand(B, L, C, 2).to(torch.cfloat)
            
            h = pscan(A_scan, X_scan) 
            
            h_delayed = torch.cat([
                torch.zeros(B, 1, C, 2, dtype=torch.cfloat, device=x.device), 
                h[:, :-1, :, :]
            ], dim=1)

            # Reconstruct audio: (B, L, 2) unsqueezed to match (B, L, C, 2)
            y_rem = (R_k.unsqueeze(2) * h_delayed).sum(dim=-1)
            y = b0_k.unsqueeze(2).real * y + y_rem.real

        return y
    









# -----------------------------------------------------------------------------




class ModelTestMC(nn.Module):

    def __init__(self, 
                 S:             int, 
                 K:             int, 
                 C:             int,
                 H:             int,
                 sampling_rate: float):
        """
        Args:
            - S             (int):    Number of cascaded processing blocks.
            - K             (int):    Number of biquad filters in each block.
            - C             (int):    Input and output channels
            - H             (int):    Channels inside the network (number of
                                      parallel filtes)
            - sampling_rate (float):  Audio sampling rate in Hz.
        """
        super().__init__()
        self.K = K
        self.S = S

        self.in_proj = nn.Linear(C, H)

        self.mid_proj = nn.Parameter(
            torch.eye(H).expand(S-1, -1, -1).clone()
        )
        self.mid_bias = nn.Parameter(torch.zeros(S-1, H))

        blocks = [MultiChannelBiquad(H=H, K=K, sampling_rate=sampling_rate) 
                  for _ in range(S)]
        self.blocks = nn.ModuleList(blocks)
        # self.gains = nn.Parameter(torch.rand(S)+0.5)
        
        self.out_proj = nn.Linear(H, C)

        self.tanh = nn.Tanh() #TanhApprox()
        

    def forward(self, y):
        if y.ndim == 2:
            y = y[:, :, None]

        y = self.in_proj(y)

        for i, block in enumerate(self.blocks):                                 # Iterate over model blocks. Each block
            # print(f"i: {i}")
            y = block(y)                                                        # consists of filter + NL
            if i != len(self.blocks)-1:
                y = y @  self.mid_proj[i] 
                y = self.tanh(y+ self.mid_bias[i])
        # print("end")

        y = self.out_proj(y)
        return y




class MultiChannelBiquad(nn.Module):
    def __init__(self, H, K, sampling_rate):
        super().__init__()
        self.H = H
        self.K = K
        self.sampling_rate = sampling_rate

        # Initialize parameters for EVERY channel independently!
        # Shape: (Channels, K)
        f_raw = torch.randn(H, K)
        
        # We can apply your safe init across the first filter of each channel
        f_start = 50 + f_raw[:, 0] * 20 
        val = torch.clamp(f_start * 4 * K / sampling_rate - 1, min=-0.99, max=0.9)
        f_raw[:, 0] = torch.asin(val)
        
        self.f_raw = nn.Parameter(f_raw)
        self.db_gain = nn.Parameter(7 * (torch.rand(H, K) - 0.5))
        self.Q_raw = nn.Parameter(torch.rand(H, K))

        self.f_raw._no_weight_decay   = True
        self.db_gain._no_weight_decay = True
        self.Q_raw._no_weight_decay   = True
        
        # other values
        Q_max = torch.ones_like(self.Q_raw)*3
        Q_max[:, 0] = Q_max[:, -1] = 1
        self.register_buffer('Q_max', Q_max)

        self.b_0 = None
        self.b_1 = None
        self.b_2 = None

        self.a_1 = None
        self.a_2 = None
    

    def forward_pscan(self, x):        
        B, L, C = x.shape
        y = x.clone()
        
        # 0. CAST TO COMPLEX (float32)
        # Parameters are natively (C, K), so no need to squeeze!
        a1 = self.a_1.to(torch.cfloat)
        a2 = self.a_2.to(torch.cfloat)
        b0 = self.b_0.to(torch.cfloat)
        b1 = self.b_1.to(torch.cfloat)
        b2 = self.b_2.to(torch.cfloat)

        # 1. CALCULATE BOTH EIGENVALUES (The Poles)
        discriminant = a1**2 - 4*a2
        
        sqrt_D = torch.where(torch.abs(discriminant) < 1e-12, 
                             torch.tensor(1e-12, dtype=torch.cfloat, device=x.device), 
                             torch.sqrt(discriminant))
        
        lam1 = (-a1 + sqrt_D) / 2.0
        lam2 = (-a1 - sqrt_D) / 2.0

        # 2. CALCULATE BOTH RESIDUES
        p1 = b1 - b0 * a1
        p2 = b2 - b0 * a2
        R1 = (p1 * lam1 + p2) / sqrt_D
        R2 = (p1 * lam2 + p2) / (-sqrt_D)

        # 3. PACK
        # Stack on the last dimension to get shape: (C, K, 2)
        lam = torch.stack([lam1, lam2], dim=-1).to(torch.cfloat) 
        R   = torch.stack([R1, R2], dim=-1).to(torch.cfloat)     
        b0  = b0.to(torch.cfloat)

        # 4. CASCADE THE FILTERS VIA PARALLEL SCAN
        for k in range(self.K):
            # Extract the k-th filter across ALL channels. Shape becomes (C, 2)
            lam_k = lam[:, k, :]
            R_k   = R[:, k, :]
            b0_k  = b0[:, k]
            
            # A_scan: Reshape (C, 2) -> (1, 1, C, 2) and expand to (B, L, C, 2)
            A_scan = lam_k.view(1, 1, self.H, 2).expand(B, L, self.H, 2)
            
            # X_scan: Expand input to include the 2 poles. Shape: (B, L, C, 2)
            X_scan = y.unsqueeze(-1).expand(B, L, self.H, 2).to(torch.cfloat)
            
            # Compute BOTH poles simultaneously! Output: (B, L, C, 2)
            h = pscan(A_scan, X_scan) 
            
            # Shift the sequence to the right by 1 step
            h_delayed = torch.cat([
                torch.zeros(B, 1, self.H, 2, dtype=torch.cfloat, device=x.device), 
                h[:, :-1, :, :]
            ], dim=1)

            # 5. RECONSTRUCT THE AUDIO
            # Reshape R_k to (1, 1, C, 2) to broadcast against h_delayed
            y_rem = (R_k.view(1, 1, self.H, 2) * h_delayed).sum(dim=-1)

            # Reshape b0_k to (1, 1, C) to broadcast against y
            y = b0_k.real.view(1, 1, self.H) * y + y_rem.real

        return y


    def compute_coefficients(self):
        # compute f
        f_squished = (torch.sin(self.f_raw) + 1)*0.25 
        
        # CRITICAL FIX: dim=1 forces sequential ordering inside each channel!
        f_cumsum = torch.cumsum(f_squished, dim=1)                              
        f = (self.sampling_rate / self.K) * f_cumsum

        # compute Q
        Q = self.Q_max * F.sigmoid(self.Q_raw) + 1e-6

        # get A, w_0, alpha 
        A = 10**(self.db_gain/40)
        w_0 = 2*torch.pi*f/self.sampling_rate
        alpha = 0.5*torch.sin(w_0)/Q

        # compute coefficients
        a_0 =  1 + alpha/A

        self.b_0 = (1 + alpha*A)      / a_0
        self.b_1 = -2*torch.cos(w_0)  / a_0
        self.b_2 = (1 - alpha*A)      / a_0

        self.a_1 = -2*torch.cos(w_0)  / a_0
        self.a_2 = (1 - alpha/A)      / a_0


    def forward(self, x: torch.Tensor):
        self.compute_coefficients()
  
        if x.ndim == 2:                                                         # x is B L C
            x = x[:, :, None]

        return self.forward_pscan(x)
    

