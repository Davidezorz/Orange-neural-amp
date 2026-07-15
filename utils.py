import torch
import matplotlib.pyplot as plt
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import *
from models.biquads_model import BiquadsBlock


def getDevice(device: str = None) -> str:                                       
    """Selects the best available device or verifies the requested one.
       If device is None: CUDA -> MPS -> CPU"""      
    if (device in [None, 'cuda']) and torch.cuda.is_available():                #   ╭ Device auto
        return 'cuda'                                                           # ◀─┤ detection  
    if (device in [None, 'mps']) and torch.backends.mps.is_available():         #   │
        return 'mps'                                                            #   │
    if device not in [None, 'mps']:                                             #   │
        print("From getDevice function: only 'cpu' is avaible")                 #   ╰
    return 'cpu'
    




def setupMatplotlib():
    """Make Matplotlib look fancyer"""
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.color'] = '#F9F9F9'





def numberOfparameters(model):
    n = sum([p.numel() for p in model.parameters()])
    return n





def plotWaveforms(y_in=None, y_true=None, y_pred=None, 
                  delta=None, start_at=0, end_at=None, vlines=[], 
                  show=False):
    assert delta is None or end_at is None, "Both delta and end_at were set"
    
    start = start_at or 0
    end = (start + delta) if delta is not None else None
    end = end_at if end_at is not None else end
    
    signals = [
        (y_true, 'true amped signal',      "#59FFA6", '-'),
        (y_pred, 'predicted amped signal', "#FF893B", '--'),
        (y_in,   'input signal',           "#7E20B9", '-')
    ]
    arrays = [signal for signal in signals if signal[0] is not None]
    assert arrays, "No input found"

    lengths = [arr[0].shape[-1] for arr in arrays]
    max_len = max(lengths)
    print('lengths:\n' + "\n".join([f'- {arr[0].shape[-1]} {arr[1]}' 
                                    for arr in arrays]))

    # plot arrays
    x = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(12, 7))
    for data, label, color, style in arrays:
        channel = ''
        data =  data[np.newaxis, :] if data.ndim == 1 else data
        
        for i, data_channel in enumerate(data):
            channel = f' channel {i}'
            ax.plot(x[start:end], data_channel[start:end], color=color, 
                    label=label+channel, linestyle=style)
    
    # plot vertica lines
    cmap = plt.get_cmap("viridis")
    n = len(vlines)
    for i, (x, label) in enumerate(vlines):
        color = cmap(i / (n-1)) if n > 1 else cmap(0)
        ax.axvline(x=x, color=color, linestyle='dashed', alpha=0.7, 
                   linewidth=1, label=label)

    ax.set_ylim(-1, 1)
    ax.legend()

    if show == True:
        plt.show()
    return fig, ax 




@torch.no_grad()
def testBiquadModel(model,y_out_val, 
                     _V3_DATA_INFO, path_distort):
    """function that test a BiquadsBlock"""
    assert isinstance(model, BiquadsBlock), "Test work only on BiquadsBlock"
    device = getDevice()

    with torch.no_grad():
        K=2
        f1_des = 400                                    # Hz
        f2_des = 10000                                   # Hz
        f1 = f1_des/_V3_DATA_INFO.rate * K
        f2 = f2_des/_V3_DATA_INFO.rate * K

        print(f"f in: {[f1, f2-f1]}\n")
        model.f_raw.copy_(torch.tensor([f1, f2-f1]))    # Targets f1_des Hz
        model.Q_raw.copy_(torch.tensor([0.0, 0.0]))     # Targets Q = 0.5
        model.db_gain.copy_(torch.tensor([-6.0, 6]))    # Targets +6.0 dB boost

    # 3. Compute the coefficients manually before the first pass
    model.compute_coefficients()

    print(f"\nTargeting: 1000 Hz, +6dB, Q=0.5")
    print(f"Calculated b0: {model.b_0}")
    print(f"Calculated b1: {model.b_1}")
    print(f"Calculated b2: {model.b_2}")
    print(f"Calculated a1: {model.a_1}")
    print(f"Calculated a2: {model.a_2}")
    print("-" * 30, "\n")

    
    Hw = model.calculate_transfer_function(model.b_0, model.b_1, model.b_2, 
                                           model.a_1, model.a_2)
    
    Hw_np = Hw.detach().cpu().numpy()

    nyquist = _V3_DATA_INFO.rate / 2
    x = np.linspace(0, nyquist , Hw_np.shape[-1])
    plt.plot(x, np.abs(Hw_np[0]*Hw_np[1]))
    # plt.ylim([-60, 6])
    plt.xscale('log')  
    plt.show()


    y_test = torch.tensor(y_out_val).to(device)
    y_test = y_test.transpose(-1, -2)[None, :, :]

    print(f"y_test.shape:        {y_test.shape}")

    model.train()
    y_biquad = model(y_test)
    print(f"y_biquad.shape:      {y_biquad.shape}")

    model.eval()
    y_biquad_eval = model(y_test)
    print(f"y_biquad_eval.shape: {y_biquad_eval.shape}")


    print(f"\nMSE: {F.mse_loss(y_biquad, y_biquad_eval)}\n")


    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(y_test[0, :, 0].detach().cpu().numpy(), color="#11FF11")
    ax.plot(y_biquad[0, :, 0].detach().cpu().numpy(), color="#1111FF")
    ax.plot(y_biquad_eval[0, :, 0].detach().cpu().numpy(), color="#FF1111")
    plt.show()

    print(torch.sum(torch.abs(y_biquad[0, :, 0]-y_test[0, :, 0])))


    store_audio(path_distort[:-4] + "y_biquad.wav",  
                y_biquad[0, :, 0].detach().cpu().numpy(), 
                _V3_DATA_INFO.rate)








