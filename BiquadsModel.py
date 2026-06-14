"""
Imprelementation of https://arxiv.org/pdf/2103.08709 without conditioning by 
https://github.com/Davidezorz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba2 import Mamba2
# from mambapy.pscan import pscan
from pscan import pscan
import math


def str_arr(x):
    res = ""
    for item in x:
        res = res + " " + f"{item.item(): .4f}"
    return res






class TanhApprox(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(1 + x**2)






class BiquadsModel(nn.Module):
    
    def __init__(self, 
                 S:             int, 
                 K:             int, 
                 sampling_rate: float, 
                 train_mode:    str | None = None, 
                 eval_mode:     str | None = None):
        """
        Neural audio processor composed of multiple cascaded biquad filter blocks
        separated by nonlinearities. Each block contains a learnable cascade of
        parametric biquad filters whose coefficients are optimized end-to-end.

        The filtering backend can be selected independently for training and
        evaluation (e.g. parallel scan, FFT, or sequential filtering).

        Args:
            - S             (int):    Number of cascaded processing blocks.
            - K             (int):    Number of biquad filters in each block.
            - sampling_rate (float):  Audio sampling rate in Hz.
            - train_mode    (str):    Filtering implementation used during 
                                      training ('ssm', 'fft', 'sfft', 'seq').
            - eval_mode     (str):    Filtering implementation used during 
                                      evaluation ('ssm', 'fft', 'sfft', 'seq').
        """
        super().__init__()
        self.K = K
        self.S = S
        self.train_mode = train_mode                                            # choose the training and
        self.eval_mode  = eval_mode                                             # evaluation computation

        # self.delay_layer = ParametricDelay()                                  # present in the paper but not so useful

        blocks = [BiquadsBlock(K=K, sampling_rate=sampling_rate) 
                  for _ in range(S)]
        self.blocks = nn.ModuleList(blocks)
        self.gains = nn.Parameter(torch.rand(S)+0.5)

        self.tanh = nn.Tanh() #TanhApprox()


    def setup_forward_mode(self):
        for block in self.blocks:
            if self.train_mode is not None: 
                block.train_mode = self.train_mode
            if self.eval_mode is not None: 
                block.eval_mode = self.eval_mode
        

    def forward(self, y):
        self.setup_forward_mode()

        if y.ndim == 2:
            y = y[:, :, None]

        # y = self.delay_layer(y)

        for i, block in enumerate(self.blocks):                                 # Iterate over model blocks. Each block
            # print(f"i: {i}")
            y = self.gains[i]*block(y)                                          # consists of filter + NL
            if i != len(self.blocks)-1:
                y = self.tanh(y)
        # print("end")
        return y





class ParametricDelay(nn.Module):
    def __init__(self):
        super().__init__()
        self.d    = nn.Parameter(torch.tensor(0.0))                             # d controls the sub-sample shift (0.0 to 1.0)
        self.gain = nn.Parameter(torch.tensor(1.0))                             # Optional global phase inversion / linear gain

    def forward(self, x):
        B, L, C = x.shape

        d = torch.sigmoid(self.d)                                               # Constrain d to be strictly between 0 and 1
        zero_pad = torch.zeros_like(B, 1, C)
        x_delayed = torch.cat([zero_pad, x[:, :-1, :]], dim=1)                  # Create a 1-sample delayed version of x
        y = (1.0 - d) * x + d * x_delayed                                       # Interpolate
        
        return self.gain * y
    



# -----------------------------------------------------------------------------

class BiquadsBlock(nn.Module):

    def __init__(self, K, sampling_rate, train_mode='ssm', eval_mode='ssm'):
        super().__init__()
        self.modes = {'ssm':  self.forward_pscan,                               # use pscan for computing the biquads
                      'fft:': self.forward_fft,                                 # use the fft on the whole sequence
                      'sfft': self.forward_sfft,                                # use the sfft, the merge point will have artefacts
                      'seq':  self.forward_sequential                           # use biquads in standard mode using a python for loop (slow)
                      }
        
        self.train_mode = train_mode                                            # choose the training and
        self.eval_mode  = eval_mode                                             # evaluation computation

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

        # other values
        Q_max = torch.ones_like(self.Q_raw)*3
        Q_max[0] = Q_max[-1] = 1
        self.register_buffer('Q_max', Q_max)

        self.b_0 = None
        self.b_1 = None
        self.b_2 = None

        self.a_1 = None
        self.a_2 = None

        # for gradient checks
        self.Hw  = None
        self.denominator = None

    
    def forward_fft(self, x):
        B, L, C = x.shape
        
        # 1. Pad the FFT to at least 2*L to prevent circular convolution (time aliasing)
        # We round up to the nearest power of 2 for maximum GPU speed
        n_fft = 2 ** math.ceil(math.log2(L * 2))

        # 2. Global 1D FFT over the entire time dimension (dim=1)
        # We don't need to reshape channels, PyTorch handles it!
        x_fft = torch.fft.rfft(x, n=n_fft, dim=1)

        # 3. Get frequency response (using the new n_fft size)
        Hw = self.calculate_transfer_function(self.b_0, self.b_1, self.b_2, 
                                              self.a_1, self.a_2, n_fft=n_fft)
        
        if self.training:
            self.Hw = Hw
            self.Hw.retain_grad()
        
        # 4. Multiply cascaded filters in Log-Space for ultimate stability
        Hw_safe = Hw + 1e-8
        H_cascade = torch.exp(torch.sum(torch.log(Hw_safe), dim=0))

        # Reshape to broadcast across (Batch, Length, Channels)
        H_cascade = H_cascade[None, :, None]

        # 5. Apply the filter
        x_processed = x_fft * H_cascade

        # 6. Global Inverse FFT
        y_padded = torch.fft.irfft(x_processed, n=n_fft, dim=1)
        
        # 7. Crop the padding off to return exactly the original length L
        y = y_padded[:, :L, :]
        
        return y


    def forward_sfft(self, x):
        B, L, C = x.shape
        n_fft = 4096*2

        # PyTorch STFT expects shape (Batch, Length)
        x_stft_in = x.transpose(1, 2).reshape(B * C, L)

        # 1. Short-Time Fourier Transform (STFT)
        # We add pad_mode="constant" to prevent the reflection crash!
        x_stft = torch.stft(
            x_stft_in, 
            n_fft=n_fft, 
            return_complex=True, 
            pad_mode="constant", 
            center=True
        )

        # 2. Get frequency response
        Hw = self.calculate_transfer_function(self.b_0, self.b_1, self.b_2, 
                                              self.a_1, self.a_2, n_fft=n_fft)
        
        self.Hw = Hw
        if self.training:
            self.Hw.retain_grad()
        
        # Multiply across the K cascaded filters
        H_cascade = torch.prod(Hw, dim=0)
        # H_cascade = torch.exp(torch.sum(torch.log(Hw + 1e-8), dim=0))

        # Reshape to (1, num_bins, 1) to broadcast across (B*C, num_bins, num_frames)
        H_cascade = H_cascade[None, :, None]

        # 3. Apply filter in frequency domain
        x_processed = x_stft * H_cascade

        # if torch.isnan(x_stft).any():       print("x_stft           CREATED NaNs!")
        # if torch.isnan(Hw).any():           print("Hw               CREATED NaNs!")
        # if torch.isnan(H_cascade).any():    print("H_cascade        CREATED NaNs!")
        # if torch.isnan(x_processed).any():  print("x_processed      CREATED NaNs!")

        # 4. Inverse STFT
        # We must explicitly tell istft that center=True so it un-pads perfectly
        y_out = torch.istft(
            x_processed, 
            n_fft=n_fft, 
            length=L, 
            center=True       
        )
        
        # Rearrange back to (B, L, C)
        y = y_out.view(B, C, L).transpose(1, 2)
        
        return y


    def forward_sequential(self, x):
        B, L, C = x.shape
    
        x = torch.cat([torch.zeros(B, 2, C, device=x.device), x], dim=1)
        
        for k in range(self.K):
            y = torch.zeros_like(x)                                             # Reset y for each new filter in the cascade
            for i in range(2, L+2):
                b_0, b_1, b_2 = self.b_0[k], self.b_1[k], self.b_2[k]
                a_1, a_2 =  self.a_1[k], self.a_2[k]
                
                b = b_0*x[:, i, :] + b_1*x[:, i-1, :] + b_2*x[:, i-2, :]  
                y[:, i, :] = b - a_1*y[:, i-1, :] - a_2*y[:, i-2, :]
            x = y.clone()

        return y[:, 2:, :]
    

    def forward_pscan(self, x):        
        B, L, C = x.shape
        y = x.clone()
        
        # 0. CAST TO DOUBLE PRECISION (float64)
        # This prevents "Catastrophic Cancellation" when the roots get too close to 0
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
    

    def calculate_transfer_function(self, b0, b1, b2, a1, a2, n_fft=4096):
        device = self.db_gain.device
        b0, b1, b2 = b0[:, None], b1[:, None], b2[:, None]                      # K 1
        a1, a2     = a1[:, None], a2[:, None]                                   # K 1
        
        # 1. Create an array of frequencies w from 0 to Pi (the Nyquist limit)
        num_bins = (n_fft // 2) + 1                                             # rfft , it returns n_fft/2 + 1 bins.
        w = torch.linspace(0, math.pi, num_bins, device=device)[None, :]        # 1 num_bins
        
        # 2. Calculate the complex delay components for all frequencies at once
        # This uses Euler's formula under the hood to handle the complex numbers
        z_inv_1 = torch.exp(-1j * w)                                            # e^(-jw)
        z_inv_2 = torch.exp(-1j * 2 * w)                                        # e^(-j2w)
        
        # 3. Calculate the Numerator (the feedforward 'b' part)
        numerator = b0 + (b1 * z_inv_1) + (b2 * z_inv_2)
        
        # 4. Calculate the Denominator (the feedback 'a' part)
        denominator = 1 + (a1 * z_inv_1) + (a2 * z_inv_2)
        
        # 5. Divide the numerator array by the denominator array
        # This gives us the final complex frequency response H(w) for every bin!
        H_w = numerator / denominator                                           # K num_bins
        if self.training:
            self.denominator = denominator
            self.denominator.retain_grad()

        # if torch.isinf(numerator).any():    print("numerator        CREATED inf!")
        # if torch.isinf(denominator).any():  print("denominator      CREATED inf!")
        # if (torch.abs(denominator) < 1e-12).any():
        #     print("denominator      near 0!")

        # if torch.isnan(numerator).any():    print("numerator        CREATED NaNs!")
        # if torch.isnan(denominator).any():  print("denominator      CREATED NaNs!")

        # if torch.isinf(H_w).any():          print("H_w              CREATED inf!")
        # if torch.isnan(H_w).any():          print("H_w              CREATED NaNs!")
        return H_w


    def compute_coefficients(self):
        """https://arxiv.org/pdf/2103.08709
           https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html"""
    
        # compute f
        # f_abs = torch.abs(self.f_raw)                                           # 1. Take absolute value of raw guesses
        f_squished = (torch.sin(self.f_raw) + 1)*0.25 # torch.abs(torch.round(f_abs) - f_abs)                      # 2. Distance to nearest integer (0.0 <= outputs <= 0.5)
        f_cumsum = torch.cumsum(f_squished, dim=0)                              # 3. force sequential ordering (f0 <= f1 <= f2...)
        f = (self.sampling_rate / self.K) * f_cumsum
        # f1 =      sr/k * |\hat f_raw - f_raw|
        # f2 = f1 + sr/k * |\hat f_raw - f_raw|


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

        # Could be safer
        # self.a_1 = torch.clamp(self.a_1, min=(-0.99 - self.a_2), max=(self.a_2 + 0.99))
        # self.a_2 = torch.clamp(self.a_2, min=-0.99, max=0.99)

        # f_tresh = 30
        # if (torch.abs(f) < f_tresh).any():
        #     print(f"f                less then {f_tresh}!")

        # if torch.isnan(self.f_raw ).any():  print("f_raw            CREATED NaNs!")
        # if torch.isnan(f ).any():           print("f                CREATED NaNs!")
        # if torch.isnan(Q ).any():           print("Q                CREATED NaNs!")

        # if (torch.abs(alpha/A) < 1e-3).any():
        #     print("alpha/A          near 0!")

        # if (torch.abs(alpha) < 1e-3).any():
        #     print("alpha            near 0!")

        # if (torch.abs(A) < 1e-3).any():
        #     print("A                near 0!")

        # print(f"alpha/A         {str_arr(alpha/A)}")
        # print(f"A               {str_arr(A)}")
        # print(f"alpha           {str_arr(alpha)}")

        # print(f"f               {str_arr(f)}")
        # print(f"self.f_raw      {str_arr(self.f_raw)}")
        # print(f"f_cumsum        {str_arr(f_cumsum)}")
        # print(f"Q               {str_arr(Q)}")
        # print(f"self.db_gain    {str_arr(self.db_gain)}")

        # print(f"self.b_0        {str_arr(self.b_0)}")
        # print(f"self.b_1        {str_arr(self.b_1)}")
        # print(f"self.b_2        {str_arr(self.b_2)}")
        # print(f"self.a_1        {str_arr(self.a_1)}")
        # print(f"self.a_2        {str_arr(self.a_2)}")


        if self.training:
            self.b_0.retain_grad()
            self.b_1.retain_grad()
            self.b_2.retain_grad()
            self.a_1.retain_grad()
            self.a_2.retain_grad()


    def forward(self, x: torch.Tensor):
        self.compute_coefficients()
  
        if x.ndim == 2:                                                         # x is B L C
            x = x[:, :, None]

        if self.training:
            return self.modes[self.train_mode](x)
        else:
            return self.modes[self.eval_mode](x)


