import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_model import LightningModel

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from preprocessing import *
from utils import plotWaveforms, setupMatplotlib, numberOfparameters, getDevice
from data import AudioDataset

from models.StateSpaceModels import Model
from models.mamba2 import Mamba2
from models.LSTM import SimpleAmpLSTM
from models.biquads_model import BiquadsBlock, BiquadsModel
from models.s4d import ModelS4D
from models.model_test import ModelTest, ModelTestMC

import warnings





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                   Main                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    print('main run')
    setupMatplotlib()

    path_data = ".data/"
    path_clean   = path_data + 'T3K-sweep-v3.wav'
    path_distort = path_data + 'v3_0_0 Sparkle Combo Distort.wav'

    y_clean, sampling_rate = load_audio(path_clean)
    y, sampling_rate       = load_audio(path_distort, sampling_rate, mono=True)
    store_audio(path_distort[:-4] + "RESAMPLED.wav",  y*.97, sampling_rate)

   
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
                                  input_mono=True, output_mono=True)

    print(gearAlignment.input_file.path)

    
    y_input, y_output  = preprocessing(gearAlignment)
    norm_factor = preprocessing.get_normalization_factor(y_output)
    y_output *= norm_factor



    gearAlignment.print_state()
    #gearAlignment.plot_alignment()








    torch.manual_seed(0)
    device = getDevice()
    #device = 'cpu'
    print(f"Using device: {device}\n")

    d_model     = 16
    d_state     = 16
    chunk_size  = 16
    headdim     = 4
    ngroups     = 4


    B, T, d_model = 2, 128, d_model

    x = torch.rand(B, T, 1).to(device)
    sr = _V3_DATA_INFO.rate
    # model = Model(H=d_model, N=d_state, D=5).to(device)
    # model = SimpleAmpLSTM(hidden_size=d_state).to(device)
    # model = BiquadsBlock(K=2, sampling_rate=_V3_DATA_INFO.rate).to(device); model.compute_coefficients()
    # model = BiquadsModel(S=8, K=4, sampling_rate=_V3_DATA_INFO.rate).to(device)
    # model = Model(H=d_model, N=d_state, D=7, input_channels=2).to(device) 
    # model = ModelS4D(N=12, H=12, D=4)
    # model = ModelTest(S=8, K=4, sampling_rate=_V3_DATA_INFO.rate).to(device)
    model = ModelTestMC(S=10, K=8, C=1, H=8, sampling_rate=sr).to(device)
    
    print('Model with: ', numberOfparameters(model), " parameters")
    # y = model(x)
    # print(y.shape) 

    file = ".weights/BiquadsS8K4"
    # model.load_state_dict(torch.load(file, weights_only=True), strict=False)


   # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --


    delay = gearAlignment.state.delay
    in_offset = -delay if delay < 0 else 0
    out_offset = delay if delay > 0 else 0
    y_in_train,  y_in_val  = preprocessing.split_train_val(y_input, 
                                            _V3_DATA_INFO,
                                            in_offset)
    y_out_train, y_out_val = preprocessing.split_train_val(y_output, 
                                            _V3_DATA_INFO, 
                                            out_offset)


    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --
    #testBiquadModel(model, _V3_DATA_INFO, path_distort)
    #raise StopIteration




    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --


    # 2. Instantiate Datasets & Dataloaders
    chunk_size = 2**14 # 2**13
    warmup = 512   
    max_epochs = 12
    batch_size = 8

    #print(y_in_val.shape)
    #print(y_out_val.shape)
    #y_in_val, y_out_val = y_in_val[:, :90000], y_out_val[:, :90000]

    train_dataset = AudioDataset(y_in_train, y_out_train, 
                                 chunk_size=chunk_size, stride=chunk_size//2)
    val_dataset   = AudioDataset(y_in_val, y_out_val, 
                                 chunk_size=y_out_val.shape[-1]) # len(x) - nx + 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4, 
                              persistent_workers=True)
    val_loader  = DataLoader(val_dataset, batch_size=1, 
                              shuffle=False, num_workers=4,
                              persistent_workers=True)

    print(f"len(train_loader): {len(train_loader)}") 
    print(f"len(val_loader):   {len(val_loader)}") 


    for y_in, y_out in val_loader:
        print(y_in.shape)
    

    
    
    lightning_model = LightningModel( 
        model          = model,
        learning_rate  = 8e-3,
        warmup         = warmup,
        lr_decay_steps = len(train_loader)*max_epochs 
    )
    # 4. Setup Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_esr',
        dirpath='.weights/',
        filename='Biquad-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    
   
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,   
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10, 
        # gradient_clip_val=1.0 #TODO: remove it
    )
    
    # 5. Train
    trainer.fit(lightning_model, train_loader, val_loader)

    # torch.save(model.state_dict(), ".weights/BiquadsS8K4")


    # 6. Train end
    lightning_model.to(device)
    lightning_model.eval()
    



    # store and show the first batch in the val_loader (it should contain only 
    # one batch)
    for i, (y_in, y_out) in enumerate(val_loader):
        y_in, y_out = y_in.to(device), y_out.to(device)

        y_pred = lightning_model.model(y_in.clone())
        y_pred_np = (y_pred[0, :, 0]).detach().cpu().numpy()

        store_audio(path_data + "last_run.wav",  
                    y_pred_np, sampling_rate)
        
        print(f"val ESR: {lightning_model.esr_loss(y_pred, y_out)}")

        plotWaveforms(y_in   = (y_in[0, :, 0]).detach().cpu().numpy(), 
                      y_true = (y_out[0, :, 0]).detach().cpu().numpy(),
                      y_pred = y_pred_np)
        plt.show()
        break


    # if we are using a biquad base model, we print the parameters.
    try:
        print()
        print(model.gains)
        for i, block in enumerate(model.blocks): 
            print()
            print()
            print(f"i: {i}")
            print(f"Calculated b0: mean: {block.b_0.mean()} values:      {block.b_0}")
            print(f"Calculated b1: mean: {block.b_1.mean()} values:      {block.b_1}")
            print(f"Calculated b2: mean: {block.b_2.mean()} values:      {block.b_2}")
            print(f"Calculated a1: mean: {block.a_1.mean()} values:      {block.a_1}")
            print(f"Calculated a2: mean: {block.a_2.mean()} values:      {block.a_2}")

            print()
            print(f"Calculated db_gain: mean: {block.db_gain.mean()}  values: {block.db_gain}")
            print(f"Calculated f_raw:   mean: {block.f_raw.mean()}    values: {block.f_raw}")
            print(f"Calculated Q_raw:   mean: {block.Q_raw.mean()}    values: {block.Q_raw}")
    except:
        pass
    """
    plotWaveforms(y, 
                  start_at=blip_locations[0]-1000, 
                  end_at=blip_locations[0]+10000,
                  vlines_list=[blip_locations[0], blip_locations[0]+delay])

    """

