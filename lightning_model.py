import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



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


    def forward(self, x):
        return self.model(x)                                                    # B T C


    def esr_loss(self, preds, targets, eps=1e-7):
        """ Error-to-Signal Ratio (ESR) loss. """

        mse = torch.mean((preds - targets)**2)
        energy = torch.mean(targets ** 2)
        return mse / (energy + eps)



    def shared_step(self, batch, batch_idx):
        y_input, y_output = batch
        B, T, C = y_output.shape

        y_pred = self(y_input)[:, -T:, :]                                       # Prediction and receptive field alignment

        loss = self.esr_loss(y_pred  [:, self.warmup:, :], 
                             y_output[:, self.warmup:, :])
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_step=True, 
                 on_epoch=True)
        

        # Log both to the progress bar simultaneously
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


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
            end_factor=0.01,                        # Decay down to 1% of self.lr
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