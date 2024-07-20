import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric import nn as geom_nn
from torch.optim import Adam


class GNNResModel(nn.Module):
    def __init__(self, input_features, edge_features, out_targets, hidden_size, fcn_hidden_dim, embedding_dim,
                 gnn_output_dim, heads):
        super().__init__()
        self.layers = []

        last_hs = input_features
        all_hss = 0

        for hs in hidden_size:
            self.layers += [
                geom_nn.GATConv(last_hs, hs, edge_dim=edge_features, heads=heads, concat=False),
                nn.ReLU(),
            ]

            last_hs = hs
            all_hss += hs

        self.layers.append(geom_nn.GCNConv(last_hs, gnn_output_dim))
        all_hss += gnn_output_dim

        self.layers = nn.ModuleList(self.layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(
                input_features + all_hss, fcn_hidden_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(fcn_hidden_dim),
            nn.Linear(fcn_hidden_dim, embedding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, out_targets),
        )

    def get_embedding(self, x, edge_index, edge_attr):
        each_step = [x]

        out = x

        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if isinstance(layer, geom_nn.GATConv):
                    out = layer(out, edge_index, edge_attr)
                else:
                    out = layer(out, edge_index)
            else:
                out = layer(out)
                each_step.append(out)

        each_step.append(out)

        out = torch.cat(each_step, dim=1)
        out = self.fcn_head(out)

        return out

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.get_embedding(x, edge_index, edge_attr)

        out = geom_nn.global_mean_pool(out, batch)
        out = self.decoder(out)

        return out


class GNN(pl.LightningModule):
    def __init__(self, model_type, model_params, recalc_mae, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        model_dict = {
            'Res': GNNResModel,
            'PNA_tg': geom_nn.PNA,
            'GAT_tg': geom_nn.GAT,
            'GCN_tg': geom_nn.GraphSAGE,
        }

        if model_type in model_dict:
            self.model = model_dict[model_type](**model_params)
        else:
            raise ValueError(f'Model type {model_type} not supported')

        self.model_type = model_type
        if model_type in ["GCN_tg", "GAT_tg", 'FiLM']:
            self.classifier = nn.Sequential(
                nn.Linear(model_params['out_channels'], 256),
                nn.Tanh(),
                nn.Linear(256, 8)
            )

        self.loss = nn.MSELoss()
        self.recal_mae = recalc_mae

    def get_embedding(self, x, edge_index, edge_attr):
        if self.model_type in ['GCN_tg', 'FiLM']:
            return self.model(x, edge_index)
        if self.model_type == 'GAT_tg':
            return self.model(x, edge_index, edge_attr)

        return self.model.get_embedding(x, edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr, batch):
        if self.model_type in ['GCN_tg', 'FiLM']:
            out = self.model(x, edge_index)
            out = geom_nn.global_mean_pool(out, batch)
            out = self.classifier(out)

            return out
        elif self.model_type == 'GAT_tg':
            out = self.model(x, edge_index, edge_attr)
            out = geom_nn.global_mean_pool(out, batch)
            out = self.classifier(out)

            return out
        else:
            return self.model(x, edge_index, edge_attr, batch)

    def log_scalar_dict(self, metrics, prefix):
        for k, v in metrics.items():
            self.log(
                f'{prefix}_{k}',
                v
            )

    def log_maes(self, y_true, y_pred, predix):
        for i in range(y_true.shape[1]):
            self.log(
                f'{predix}_mae_{i}',
                torch.mean(torch.abs(y_true[:, i] - y_pred[:, i])).item() / self.recal_mae[i]
            )

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch.y
        )

        self.log('train_loss', loss)

        # Compute MSE for individual targets
        for i in range(y.shape[1]):
            self.log(f'train_loss_{i}', self.loss(y[:, i], batch.y[:, i]))

        self.log_maes(batch.y, y, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch.y
        )

        self.log('val_loss', loss)

        # Compute MSE for individual targets
        for i in range(y.shape[1]):
            self.log(f'val_loss_{i}', self.loss(y[:, i], batch.y[:, i]))

        self.log_maes(batch.y, y, 'val')

        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch.y
        )

        self.log('test_loss', loss)

        # Compute MSE for individual targets
        for i in range(y.shape[1]):
            self.log(f'test_loss_{i}', self.loss(y[:, i], batch.y[:, i]))

        self.log_maes(batch.y, y, 'test')

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
