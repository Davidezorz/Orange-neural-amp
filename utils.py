import torch
import matplotlib.pyplot as plt
import datetime
import numpy as np


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











