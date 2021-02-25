import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from GTNmodel import GTN, GTNconfig
from GNNBenchmarkDataModule import GNNBenchmarkDataModule


class GTNNodeClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.model = GTN(config)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)

        J = self.loss(logits, y)

        logits = torch.nn.Softmax(dim=1)(logits)
        acc = accuracy(logits, y)

        return {"loss": J, "accuracy": acc}

    def training_epoch_end(self, train_step_outputs):
        mean_loss = torch.Tensor([loss["loss"]
                                  for loss in train_step_outputs]).mean()
        mean_acc = torch.Tensor([acc["accuracy"]
                                 for acc in train_step_outputs]).mean()
        self.log("train loss", mean_loss, prog_bar=True)
        self.log("train acc", mean_acc, prog_bar=True)
        #print("train loss:", mean_loss.item(), "train accuracy:",  mean_acc.item())

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        mean_loss = torch.Tensor([loss["loss"]
                                  for loss in val_step_outputs]).mean()
        mean_acc = torch.Tensor([acc["accuracy"]
                                 for acc in val_step_outputs]).mean()
        self.log("val acc", mean_acc, prog_bar=True)
        #print("validation loss:", mean_loss.item(), "validation accuracy:",  mean_acc.item())

    """
    def train_dataloader(self):
        dataset = GNNBenchmarkDataset(root="/tmp/", name="CLUSTER")
        for data in dataset:
            data.edge_index = add_self_loops(data.edge_index)[0]
        train_loader = DataLoader(dataset[:1000], batch_size=2, shuffle=True)

        return train_loader
      
    def val_dataloader(self):
        dataset = GNNBenchmarkDataset(root="/tmp/", name="CLUSTER", split="val")
        for data in dataset:
            data.edge_index = add_self_loops(data.edge_index)[0]
        val_loader = DataLoader(dataset[:500], batch_size=256)

        return val_loader
        """


if __name__ == "__main__":
    # datamodule
    dm = GNNbenchmarkDataModule("CLUSTER")
    dm.prepare_data()
    dm.setup('fit')

    # init model
    conf_dict = {'embedding_dim': 128, 'num_heads': 4, 'attn_pdrop': 0.0, 'resid_pdrop': 0.0,
                 'num_layers': 5, 'num_classes': dm.num_classes, 'input_dim': dm.num_features}
    conf = GTNconfig(**conf_dict)
    model = GTNNodeClassifier(conf)

    trainer = pl.Trainer(
        max_epochs=10, progress_bar_refresh_rate=10, gradient_clip_val=0.1)
    trainer.fit(model)
