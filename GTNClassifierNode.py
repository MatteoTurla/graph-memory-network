import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix
from GTNmodel import GTN, GTNconfig
from GNNBenchmarkDataModule import GNNBenchmarkDataModule


class GTNNodeClassifier(pl.LightningModule):

    def __init__(self, conf_dict):
        super().__init__()

        self.initial_lr = self.conf_dict["initial_lr"]

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
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=5)
        return {
            "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"
        }
   

    def training_step(self, batch, batch_idx):
        y = batch.y

        # randomly flip sign as in gnnbenchmark paper
        pos_enc = batch.pos_enc
        sign_flip = torch.rand(pos_enc.size(1), device=pos_enc.device, requires_grad=False)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        batch.pos_enc = pos_enc * sign_flip.unsqueeze(0)

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return J

    def validation_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('val_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('test_loss', J, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {"logits": logits, "y": y}

    def test_epoch_end(self, results):
        logits = torch.cat([result["logits"] for result in results])
        y = torch.cat([result["y"] for result in results])

        print(confusion_matrix(logits, y, self.num_classes))