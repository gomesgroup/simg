import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv import PNAConv
from torch.optim.lr_scheduler import ReduceLROnPlateau

LP_N_CLASSES = 5
N_ATOM_FEATURES = 16


class GNN_LP_Model(nn.Module):
    def __init__(self, hidden_size, deg):
        super().__init__()
        self.layers = []

        last_hs = N_ATOM_FEATURES

        self.aggregators = ["sum", "mean", "max", "min", "std"]
        self.scalers = ["identity"]

        for hs in hidden_size:
            self.layers += [
                PNAConv(
                    last_hs, hs, self.aggregators, self.scalers, deg=deg, edge_dim=4
                ),
                nn.ReLU(),
            ]

            last_hs = hs

        self.layers.append(GCNConv(last_hs, N_ATOM_FEATURES))
        self.layers = nn.ModuleList(self.layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(N_ATOM_FEATURES * 2, 10), nn.ReLU(), nn.Linear(10, LP_N_CLASSES * 2)
        )

    def forward(self, x, edge_index, edge_attr):
        out = x

        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                if isinstance(layer, PNAConv):
                    out = layer(out, edge_index, edge_attr)
                else:
                    out = layer(out, edge_index)
            else:
                out = layer(out)

        out = torch.cat((out, x), dim=1)
        out = self.fcn_head(out)

        return out


class GNN_LP(LightningModule):
    def __init__(self, config, deg):
        super().__init__()

        hidden_size = config["hidden_size"]
        self.lr = config["lr"]
        self.wd = config["wd"]
        self.batch_size = config["batch_size"]

        self.model = GNN_LP_Model(hidden_size, deg)

        self.aggregators = self.model.aggregators
        self.scalers = self.model.scalers

        self.loss1 = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        return self.model(x, edge_index, edge_attr)

    def custom_loss(self, y_hat, y):
        y_hat_1 = y_hat[:, :LP_N_CLASSES]
        y_1 = y[:, 0]

        y_hat_2 = y_hat[:, LP_N_CLASSES:]
        y_2 = y[:, 1]

        loss1 = self.loss1(y_hat_1, y_1)
        loss2 = self.loss1(y_hat_2, y_2)

        self.log("#lps_loss", loss1.item(), batch_size=self.batch_size)
        self.log("#diff_lps_loss", loss2.item(), batch_size=self.batch_size)

        return loss1 + loss2 * 2

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        y = batch.y

        loss = self.custom_loss(y_hat, y)
        self.log("train_loss", loss.item(), batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        y = batch.y

        loss = self.custom_loss(y_hat, y)
        self.log("val_loss", loss.item(), batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        y = batch.y

        loss = self.custom_loss(y_hat, y)
        self.log("test_loss", loss.item(), batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.75, patience=3  # , verbose=True
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]
