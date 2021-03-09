import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.functional import confusion_matrix
from GNN import GNN, GNNconfig
from GNNBenchmarkDataModule import GNNBenchmarkDataModule


class GTNNodeClassifier(pl.LightningModule):

    def __init__(self, conf_dict):
        super().__init__()

        # pass to optimizer
        self.initial_lr = conf_dict["initial_lr"]

        # define the model
        config = GNNconfig(**conf_dict)
        self.model = GTN(config)

        # metric to log
        self.metric = Accuracy()

        # define loss
        self.loss = torch.nn.CrossEntropyLoss()

        # save the configutation dictionary
        self.save_hyperparameters(conf_dict)

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=5)
        return {
            "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"
        }
   

    def training_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        y_pred = torch.nn.Softmax(dim=1)(logits)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_acc', self.metric(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return J

    def validation_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        y_pred = torch.nn.Softmax(dim=1)(logits)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('val_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_acc', self.metric(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        y_pred = torch.nn.Softmax(dim=1)(logits)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('test_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('test_acc', self.metric(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {"logits": logits, "y": y}

    def test_epoch_end(self, results):
        logits = torch.cat([result["logits"] for result in results])
        y = torch.cat([result["y"] for result in results])

        print(confusion_matrix(logits, y, self.num_classes))