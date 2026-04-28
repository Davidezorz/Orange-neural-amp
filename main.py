import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_model import LightningModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from preprocessing import *
from utils import plotWaveforms, setupMatplotlib, numberOfparameters, getDevice
from mamba2 import Mamba2
from model import Model
from data import AudioDataset
from LSTM import SimpleAmpLSTM

import warnings
import logging

if __name__ == "__main__":
    print('main run')
    setupMatplotlib()

    path_clean   = '.data/T3K-sweep-v3.wav'
    path_distort = '.data/v3_0_0 Sparkle Combo Distort.wav'

    y_clean, sampling_rate = load_audio(path_clean)
    y, sampling_rate       = load_audio(path_distort, sampling_rate, mono=True)
    store_audio(path_distort[:-4] + "RESAMPLED.wav",  y*.97, sampling_rate)

    raise Exception
    # audio_data = audio_data.mean(axis=0)
    print(y_clean.shape, sampling_rate)


    # V3:
    # (0:00-0:09) Validation 1
    # (0:09-0:10) Silence
    # (0:10-0:12) Blips at 0:10.5 and 0:11.5
    # (0:12-0:15) Chirps
    # (0:15-0:17) Noise
    # (0:17-3:00.5) General training data
    # (3:00.5-3:01) Silence
    # (3:01-3:10) Validation 2
    _V3_DATA_INFO = DataInfo(
        rate                = 48000,
        validation_section1 = (0,         9*48000),
        validation_section2 = (181*48000, 190*48000),
        blip_section        = (480_000,   576_000),
        blip_locations      = (504_000,   552_000),
        background_interval = (492_000,   498_000),
    )

    # audio_data[480_000:576_000]
    # plotWaveforms(y_clean[503_990:504_050], y[503_990:504_050])
    preprocessing = Preprocessing(lookahead=5000)
    gearAlignment = CapturePair(_V3_DATA_INFO, path_distort, path_clean,
                                  input_mono=True, output_mono=False)

    print(gearAlignment.input_file.path)

    
    y_input, y_output  = preprocessing(gearAlignment)
    norm_factor = preprocessing.get_normalization_factor(y_output)
    y_output *= norm_factor



    gearAlignment.print_state()
    #gearAlignment.plot_alignment()








    torch.manual_seed(0)
    device = getDevice()
    print(f"Using device: {device}\n")

    d_model     = 16
    d_state     = 32
    chunk_size  = 16
    headdim     = 4
    ngroups     = 4


    B, T, d_model = 2, 128, d_model
    """
    x = torch.rand(B, T, d_model).to(device)
    torch.manual_seed(0)
    model = Mamba2(d_model,
                    d_state               = d_state, 
                    headdim               = headdim, 
                    chunk_size            = chunk_size,
                    expand                = 2,
                    ngroups               = ngroups,
                    learnable_init_states = True
                    ).to(device)
    """

    x = torch.rand(B, T, 1).to(device)
    model = Model(H=d_model, N=d_state, D=8).to(device)
    # model = SimpleAmpLSTM(hidden_size=d_state).to(device)
    
    print('Model with: ', numberOfparameters(model), " parameters")
    y = model(x)
    print(y.shape) 
   


    delay = gearAlignment.state.delay
    in_offset = -delay if delay < 0 else 0
    out_offset = delay if delay > 0 else 0
    y_in_train,  y_in_val  = preprocessing.split_train_val(y_input, 
                                            _V3_DATA_INFO,
                                            in_offset)
    y_out_train, y_out_val = preprocessing.split_train_val(y_output, 
                                            _V3_DATA_INFO, 
                                            out_offset)


    # 2. Instantiate Datasets & Dataloaders
    chunk_size = 2**14
    warmup = 0   
    max_epochs = 30
    batch_size = 8


    train_dataset = AudioDataset(y_in_train, y_out_train, 
                                 chunk_size=chunk_size)
    val_dataset   = AudioDataset(y_in_val, y_out_val, 
                                 chunk_size=chunk_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4, 
                              persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4,
                              persistent_workers=True)

    print(f"len(train_loader): {len(train_loader)}") 
        

    
    lightning_model = LightningModel(
        model          = model,
        learning_rate  = 8e-4,
        warmup         = warmup,
        lr_decay_steps = len(train_loader)*max_epochs 
    )
    # 4. Setup Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='.weights/',
        filename='mamba2-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    
   
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,   # Automatically uses GPU if available
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    # 5. Train
    trainer.fit(lightning_model, train_loader, val_loader)


    # 6. Train end
    lightning_model.to(device)
    lightning_model.eval()
    
    """
    checkpoint_path = ".weights/mamba2-epoch=29-val_loss=0.6456.ckpt"

    lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model 
    )
    

    """
    """
    losses = []
    for i, (y_in, y_out) in enumerate(val_loader):
        y_in, y_out = y_in.to(device), y_out.to(device)

        y_pred = lightning_model.model(y_in)
        loss = lightning_model.esr_loss(y_pred, y_out)
        losses.append(loss.item())
        if i == 5: 
            for j , (y_in_b, y_out_b, y_pred_b) in enumerate(zip(y_in, y_out, y_pred)):
                print(f"iteration {j} current loss: ", lightning_model.esr_loss(y_pred_b, y_out_b))
                plotWaveforms(y_in   = (y_in_b[:, 0]).detach().cpu().numpy(), 
                              y_true = (y_out_b[:, 0]).detach().cpu().numpy(), 
                              y_pred = (y_pred_b[:, 0]).detach().cpu().numpy())
                plt.show()

    worst_batch = torch.argmax(torch.tensor(losses))
    print()
    print(f"worst batch {worst_batch}")
    print(f"worst batch {torch.tensor(losses)[worst_batch]}")
    print(f"val loss {sum(losses)/len(losses)}")
    print()
    """

    
    for i, (y_in, y_out) in enumerate(val_loader):
        y_in, y_out = y_in.to(device), y_out.to(device)
        if i >= 5 : break

        y_pred = lightning_model.model(y_in)
        print(lightning_model.esr_loss(y_pred, y_out))

        
        plotWaveforms(y_in   = (y_in[0, :, 0]).detach().cpu().numpy(), 
                      y_true = (y_out[0, :, 0]).detach().cpu().numpy(), 
                      y_pred = (y_pred[0, :, 0]).detach().cpu().numpy())
        plt.show()


    """
    plotWaveforms(y, 
                  start_at=blip_locations[0]-1000, 
                  end_at=blip_locations[0]+10000,
                  vlines_list=[blip_locations[0], blip_locations[0]+delay])

    """