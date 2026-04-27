import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing import DataInfo






class AudioDataset(Dataset):
    """
    A streamlined dataset for Neural Amp Modeler.
    Assumes `y_input` and `y_output` are 2D numpy arrays of the same length and
    sample rate.
    """
    def __init__(
        self,
        y_input:            np.ndarray,
        y_output:           np.ndarray,
        chunk_size:         int     = None,
        stride:             int     = None,
        receptive_field:    int     = 1,
    ):
        self._checkArgs(y_input, y_output)
        channels_in,  len_in  = y_input.shape

        self.y_input  = torch.from_numpy(y_input.T).float().contiguous()        # Transpose!
        self.y_output = torch.from_numpy(y_output.T).float().contiguous()       # We will have B T Channels
        
        max_amp = torch.abs(self.y_output).max()
        if max_amp > 1.0:
            raise ValueError(f"Output target is clipping (amplitude >= 1.0) " \
                             f"after scaling, amplitude: {max_amp}")
            
        self.receptive_field = receptive_field
        self.stride = stride if stride is not None else chunk_size
        self.chunk_size = chunk_size if chunk_size is not None \
                          else len_in - receptive_field + 1
        
        if self.receptive_field > len_in:
            raise ValueError(f"Audio length ({len_in}) is too short for" \
                             f"receptive field ({self.receptive_field}).")
        
    
    def _checkArgs(self, y_input, y_output):
        if y_input.ndim != y_output.ndim != 2:
            raise ValueError("args should be a 2D array")

        channels_in,  len_in  = y_input.shape
        channels_out, len_out = y_output.shape

        if len_in != len_out:
            raise ValueError(f"Length mismatch: input {len_in}, out {len_out}")
        

    def __len__(self) -> int:
        """Compute how many chunks fit into the audio based on the stride."""
        len_in, channels_in  = self.y_input.shape
        available_output_samples = len_in - self.receptive_field + 1
        
        if available_output_samples < self.chunk_size: return 0                 # Compute how many strides 
        return 1 + (available_output_samples - self.chunk_size) // self.stride  # fit into that space


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.stride
        j = i + self.chunk_size
        context_offset = self.receptive_field - 1
    
        in_chunk  = self.y_input [i                  : j + context_offset, :]   # Input needs to include the past
        out_chunk = self.y_output[i + context_offset : j + context_offset, :]   # context plus the current chunk

        return in_chunk, out_chunk
    