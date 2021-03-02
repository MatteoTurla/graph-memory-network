import torch
import pytorch_lightning as pl
from GTNmodel import GTN, GTNconfig
from ZincDataModule import ZINCDataModule
from torch_geometric.nn import global_mean_pool

class GTNNodeClassifier(pl.LightningModule):

    def __init__(self, conf_dict):
        super().__init__()

        config = GTNconfig(**conf_dict)
        self.model = GTN(config)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(config.embedding_dim, config.embedding_dim*2)
                torch.nn.ReLU()
                torch.nn.Linear(config.embedding_dim*2, config.num_classes)
            )
        
        # define absolute and squared loss
        self.loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()

        self.save_hyperparameters(conf_dict)

    def forward(self, data):
        x =  self.model(data)
        x = global_mean_pool(x, data.batch)
        logits = self.mlp(x)

        return logits


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=5)
        return {
            "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"
        }
   
    def training_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return J

    def validation_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)
        mae = self.mae_loss(logits, y)

        self.log('val_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_mae_loss', mae, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)
        mae = self.mae_loss(logits, y)

        self.log('val_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_mae_loss', mae, on_step=False, on_epoch=True, prog_bar=True, logger=False)