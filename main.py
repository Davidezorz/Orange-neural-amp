import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_model import LightningModel
from lightning_model2 import LightningModel2

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from preprocessing import *
from utils import plotWaveforms, setupMatplotlib, numberOfparameters, getDevice
from mamba2 import Mamba2
from model import Model
from data import AudioDataset
from LSTM import SimpleAmpLSTM
from BiquadsModel import BiquadsBlock, BiquadsModel
from s4d import ModelS4D

import warnings


def testBiquadModel(model, _V3_DATA_INFO, path_distort):
    """function that test a BiquadsBlock"""
    assert isinstance(model, BiquadsBlock), "Test work only on BiquadsBlock"
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
                y_biquad[0, :, 0].detach().cpu().numpy(), sampling_rate)





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                   Main                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    print('main run')
    setupMatplotlib()

    path_clean   = '.data/T3K-sweep-v3.wav'
    path_distort = '.data/v3_0_0 Sparkle Combo Distort.wav'

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
    # model = Model(H=d_model, N=d_state, D=5).to(device)
    # model = SimpleAmpLSTM(hidden_size=d_state).to(device)
    # model = BiquadsBlock(K=2, sampling_rate=_V3_DATA_INFO.rate).to(device); model.compute_coefficients()
    model = BiquadsModel(S=10, K=12, sampling_rate=_V3_DATA_INFO.rate).to(device)
    # model = Model(H=d_model, N=d_state, D=7, input_channels=2).to(device) 
    # model = ModelS4D(N=12, H=12, D=4)
    
    print('Model with: ', numberOfparameters(model), " parameters")
    # y = model(x)
    # print(y.shape) 

    file = ".weights/BiquadsS8K4"
    # model.load_state_dict(torch.load(file, weights_only=True))


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
    chunk_size = 2**13
    warmup = 512   
    max_epochs = 12*2
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
    

    
    
    lightning_model = LightningModel(  # TODO: HERE should be LightningModel
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

        y_pred = lightning_model.model(y_in.clone())

        print(lightning_model.esr_loss(y_pred, y_out))

        plotWaveforms(y_in   = (y_in[0, :, 0]).detach().cpu().numpy(), 
                      y_true = (y_out[0, :, 0]).detach().cpu().numpy(),
                      y_pred = (y_pred[0, :, 0]).detach().cpu().numpy())
        plt.show()

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