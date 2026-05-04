import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import losses



class LightningModel(pl.LightningModule):
    def __init__(
        self, 
        model:           torch.nn.Module,
        learning_rate:   float = 1e-3,
        weight_decay:    float = 1e-4,
        warmup:          int   = 64,
        lr_decay_steps:  int   = 100
        
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  
        self.lr              = learning_rate
        self.weight_decay    = weight_decay
        self.warmup          = warmup
        self.lr_decay_steps  = lr_decay_steps
        
        self.model = model

        self.esr_loss       = losses.ESRLoss()
        self.weak_esr_loss  = losses.WeakESRLoss()
        self.mse_loss       = losses.MSELoss()
        self.MRSTFTLoss     = losses.MultiResolutionSTFTLoss()

    def forward(self, x):
        return self.model(x)                                                    # B T C


    def shared_step(self, batch, batch_idx):
        y_input, y_output = batch
        B, L, C = y_output.shape

        y_pred = self(y_input)[:, -L:, :]                                       # Prediction and receptive field alignment

        esr_loss = self.esr_loss(y_pred  [:, self.warmup:, :], 
                                 y_output[:, self.warmup:, :])
        
        weak_esr_loss = self.weak_esr_loss(y_pred  [:, self.warmup:, :], 
                                           y_output[:, self.warmup:, :])
        
        print()
        print(y_pred.transpose(-1, -2).contiguous().shape)
        print(y_output.transpose(-1, -2).contiguous().shape)
        print()
        mrSTFTLoss = self.MRSTFTLoss(y_pred.transpose(-1, -2).contiguous(), 
                                     y_output.transpose(-1, -2).contiguous())
        return esr_loss, weak_esr_loss, mrSTFTLoss


    def training_step(self, batch, batch_idx):
        esr_loss, weak_esr_loss, mrSTFTLoss = self.shared_step(batch, batch_idx)
        loss = 0.1*weak_esr_loss + 0.9*esr_loss

        # Logs
        self.log('train_esr', esr_loss, prog_bar=True, on_step=True, 
                 on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, 
                 on_epoch=True)
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        esr_loss, weak_esr_loss, mrSTFTLoss = self.shared_step(batch, batch_idx)
        self.log('val_esr', esr_loss, prog_bar=True, on_step=False, 
                 on_epoch=True)
        return esr_loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Linearly decay the learning rate over a set number of epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0,                       # Start at 100% of self.lr
            end_factor=0.05,                        # Decay down to 5% of self.lr
            total_iters=self.lr_decay_steps         # Number of epochs over which to decay
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1
            }
        }
    



    