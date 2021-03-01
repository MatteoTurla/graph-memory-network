import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from GTNmodel import GTN, GTNconfig
from GNNBenchmarkDataModule import GNNBenchmarkDataModule


import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix
from GTNmodel import GTN, GTNconfig
from GNNBenchmarkDataModule import GNNBenchmarkDataModule


class GTNNodeClassifier(pl.LightningModule):

    def __init__(self, conf_dict):
        super().__init__()

        config = GTNconfig(**conf_dict)
        self.model = GTN(config)
        self.num_classes = config.num_classes

        if conf_dict["weighted_loss"]:
            self.loss = self.weighted_loss
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters(conf_dict)

    def forward(self, data):
        return self.model(data)

    def weighted_loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[torch.nonzero(label_count)].squeeze()
        cluster_sizes = torch.zeros(self.num_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=1e-6, patience=3)
        return {
            "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"
        }

    def training_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', J, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)
        self.log('train_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        return J

    def validation_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('val_loss', J, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)
        self.log('val_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('test_loss', J, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)
        self.log('test_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        return {"logits": logits, "y": y}

    def test_epoch_end(self, results):
        logits = torch.cat([result["logits"] for result in results])
        y = torch.cat([result["y"] for result in results])

        print(confusion_matrix(logits, y, self.num_classes))


if __name__ == "__main__":
    # datamodule
    batch_size = 8
    k = 2
    dm = GNNBenchmarkDataModule("CLUSTER", batch_size=batch_size, k)
    dm.prepare_data()
    dm.setup('fit')

    # init model
    conf_dict = {'embedding_dim': 128, 'num_heads': 4, 'attn_pdrop': 0.0, 'resid_pdrop': 0.0,
                 'num_layers': 6, 'num_classes': dm.num_classes, 'input_dim': dm.num_features,
                 'weighted_loss': False, 'batch_size': batch_size, 'k_lap_ecnoding': k}

    model = GTNNodeClassifier(conf_dict)

    trainer = pl.Trainer(
        max_epochs=100, progress_bar_refresh_rate=10, gradient_clip_val=0.1, gpus=1,
        limit_train_batches=1.0, limit_val_batches=1.0)
    trainer.fit(model, dm)
