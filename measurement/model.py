
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN_Classifier(nn.Module):
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        self.L = num_classes # we use the variable name L for the number of classes in legacy of the SLADS-Net paper
        
        # Kernel size of 10, stride of 2, following 
        self.layers = nn.Sequential(
            # First conv/pooling layer
            nn.Conv1d(1, 8, kernel_size=10, stride=2), # padding_mode='replicate'
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=10, stride=2),
            
            # Second conv/pooling layer
            nn.Conv1d(8, 8, kernel_size=10, stride=2), # padding_mode='replicate'
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=10, stride=2),
            
            # FC (fully connected) layers
            nn.Flatten(),
            nn.LazyLinear(100), 
            nn.Linear(100, 32),
            nn.Linear(32, self.L),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        
        return y

class CNN_Classifier_Lightning(pl.LightningModule):
    def __init__(self, input_shape, num_classes: int = 2, lr: float = 0.001, lr_gamma: float = 0.99):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_Classifier(num_classes)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        targets = y.squeeze().long()
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        