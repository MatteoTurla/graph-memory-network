import torch
import pytorch_lightning as pl
from GTNClassifierNode import GTNNodeClassifier
from GNNBenchmarkDataModule import GNNBenchmarkDataModule

if __name__ == "__main__":
    # datamodule
    batch_size = 8
    k = 20
    dm = GNNBenchmarkDataModule(dataset_name="CLUSTER", batch_size=batch_size, 
                                k=k)
    dm.prepare_data()
    dm.setup('fit')
    # init model
    # print a summary
    # init model
    conf_dict = {'embedding_dim': 72, 
                'num_heads': 12, 
                'attn_pdrop': 0.0, 
                'resid_pdrop': 0.0,
                'num_layers': 12, 
                'num_classes': dm.num_classes, 
                'input_dim': dm.num_features,
                'batch_size': batch_size, 
                'weighted_loss': False,
                'norm':"batch", # layer or batch norm
                'final_layer': "mlp", #mlp | gtp | graph_regression
                'init_weights': "custom", # custom (gtp-3) vs default
                'pos_dim':k # number of eigenvetor in positional laplacian encoding
                }

    model = GTNNodeClassifier(conf_dict)

    model.summarize("top") # full | top

    trainer = pl.Trainer(
    max_epochs=10, progress_bar_refresh_rate=20, gradient_clip_val=0.1, gpus=1,
    limit_train_batches=1.0, limit_val_batches=1.0)
    trainer.fit(model, dm)
