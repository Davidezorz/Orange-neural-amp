import torch
import torch.nn as nn
import torch.nn.functional as F

from auraloss.auraloss.freq import STFTLoss


"""
╭ CONVENTIONS ────────────────────────────────────────────────────────────────╮
│ ├─• B     ▶ batch size                                                      │
│ ├─• L     ▶ number of samples per channel                                   │
│ ╰─• C     ▶ audio channels                                                  │ 
╰─────────────────────────────────────────────────────────────────────────────╯
"""






# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             MSE from pyTorch                              │
# ╰───────────────────────────────────────────────────────────────────────────╯

MSELoss = nn.MSELoss





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                 ESR Loss                                  │
# ╰───────────────────────────────────────────────────────────────────────────╯

class ESRLoss(nn.Module):
    """Error-to-Signal Ratio (ESR) loss."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps


    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        mse = torch.mean((preds - targets) ** 2)
        energy = torch.mean(targets ** 2)
        return mse / (energy + self.eps)






# ╭───────────────────────────────────────────────────────────────────────────╮
# │                              Weak ESR Loss                                │
# ╰───────────────────────────────────────────────────────────────────────────╯

class WeakESRLoss(nn.Module):
    """Error-to-Signal Ratio (ESR) loss with a softened denominator."""

    def __init__(self, coef: float = 0.1):
        super().__init__()
        if not (0.0 <= coef <= 1.0):
            raise ValueError("coef must be in [0, 1]")
        self.coef = coef


    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        mse = torch.mean((preds - targets) ** 2)
        energy = torch.mean(targets ** 2)
        denom = self.coef + (1 - self.coef) * energy
        return mse / denom
 




# ╭───────────────────────────────────────────────────────────────────────────╮
# │                            Pre Emphasis Filter                            │
# ╰───────────────────────────────────────────────────────────────────────────╯

class PreEmphasisFilter(nn.Module):
    """ Compute first-order pre-emphsis filter """

    def __init__(self, coef: float = 0.1):
        super().__init__()
        self.coef = coef


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """ apply the filter: y: (*, L), return: (*, L-1) """
        return y[..., 1:] - self.coef * y[..., :-1]






# ╭───────────────────────────────────────────────────────────────────────────╮
# │                   Auraloss Multi-Resolution STFT Loss                     │
# ╰───────────────────────────────────────────────────────────────────────────╯


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes:              list[int] = [1024, 2048, 512],
        hop_sizes:              list[int] = [120, 240, 50],
        win_lengths:            list[int] = [600, 1200, 240],
        window:                 str       = "hann_window",
        w_sc:                   float     = 1.0,
        w_log_mag:              float     = 1.0,
        w_lin_mag:              float     = 0.0,
        w_phs:                  float     = 0.0,
        sample_rate:            float     = None,
        scale:                  str       = None,
        n_bins:                 int       = None,
        perceptual_weighting:   bool      = False,
        scale_invariance:       bool      = False,
        **kwargs,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    perceptual_weighting,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def _reshape_args(self, y: torch.Tensor):
        if y.ndim == 1:
            return y[None, None, :]
        elif y.ndim == 2:
            return y[:, None, :]
        elif y.ndim == 3:
            return y
        raise ValueError(f"Too many dims: expected at maximum 3, got {y.ndim}")
    

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:    (B C L) or (B L) or (L)
        targets:  (B C L) or (B L) or (L)
        """
        preds   = self._reshape_args(preds)
        targets = self._reshape_args(targets)

        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(preds, targets)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(preds, targets)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
        

