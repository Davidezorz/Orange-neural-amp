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

        self.need_stop = 0


    def forward(self, x):
        return self.model(x)                                                    # B T C


    def shared_step(self, batch, batch_idx):
        y_input, y_output = batch
        B, L, C = y_output.shape

        y_pred = self(y_input)[:, -L:, :]                                       # Prediction and receptive field alignment

        #if torch.isnan(y_pred).any():
        #    print(f"!!! FORWARD SIGNAL EXPLODED AT BATCH {batch_idx} !!!")

        mse = self.mse_loss(y_pred  [:, self.warmup:, :], 
                                 y_output[:, self.warmup:, :])
        
        esr_loss = self.esr_loss(y_pred  [:, self.warmup:, :], 
                                 y_output[:, self.warmup:, :])
        
        weak_esr_loss = self.weak_esr_loss(y_pred  [:, self.warmup:, :], 
                                           y_output[:, self.warmup:, :])
        
        #print()
        #print(y_pred.transpose(-1, -2).contiguous().shape)
        #print(y_output.transpose(-1, -2).contiguous().shape)
        #print()
        mrSTFTLoss = self.MRSTFTLoss(y_pred.transpose(-1, -2).contiguous(), 
                                     y_output.transpose(-1, -2).contiguous())
        
        return esr_loss, weak_esr_loss, mrSTFTLoss, mse


    def training_step(self, batch, batch_idx):
        # print()
        # print("--"*40)
        # print(f"=== BATCH IDX: {batch_idx}")
        #if self.need_stop > 5:
        # if batch_idx > 1: raise Exception
        losses = self.shared_step(batch, batch_idx)
        esr_loss, weak_esr_loss, mrSTFTLoss, mse_loss = losses
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
    

    def on_before_optimizer_step(self, optimizer):
        return
        # Loop through every BiquadsBlock in your model
        for i, block in enumerate(self.model.blocks):
            
            # Group the coefficients together so we can loop through them easily
            coeffs = {
                'b_0': block.b_0, 'b_1': block.b_1, 'b_2': block.b_2,
                'a_1': block.a_1, 'a_2': block.a_2, 'Hw': block.Hw
            }
            #print(f"BLOCK {i}")
            #print(f"block.dn:      {block.denominator}")
            #print(f"block.dn.grad: {block.denominator.grad}")

            for name, tensor in coeffs.items():
                # Make sure the tensor and its gradient actually exist
                if tensor is not None and tensor.grad is not None:
                    
                    # Find the largest absolute gradient value in the K=10 filters
                    max_grad = torch.max(torch.abs(tensor.grad))

                    self.log(f"grad/{i}{name}", max_grad, on_step=True, on_epoch=False)
                    if max_grad > 500 or torch.isnan(max_grad): 
                        # print("\n\n")
                        # print(f"i: {i}")
                        # print(f"maxgrad: {max_grad}")
                        # print("argmax: ", torch.argmax(torch.abs(tensor.grad)))
                        # print(f"{name}: {tensor}")
                        # print(f"{name} grad: {tensor.grad}")
                        # print(f" block.db_gain: {block.db_gain} \n block.f_raw: {block.f_raw} \n block.Q_raw: {block.Q_raw}")
                        # print(f"grads: \n block.db_gain: {block.db_gain.grad} \n block.f_raw: {block.f_raw.grad} \n block.Q_raw: {block.Q_raw.grad}")

                        self.need_stop = self.need_stop + 1


    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch, batch_idx)
        esr_loss, weak_esr_loss, mrSTFTLoss, mse_loss = losses
        self.log('val_esr', esr_loss, prog_bar=True, on_step=False, 
                 on_epoch=True)
        self.log('val_mse', mse_loss, prog_bar=True, on_step=False, 
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
    



    