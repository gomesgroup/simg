from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv import PNAConv

from pytorch_lightning import LightningModule

LP_N_CLASSES = 5
N_ATOM_FEATURES_LP_MODEL = 16


class GNN_LP_Model(nn.Module):
    def __init__(self, hidden_size, deg):
        super().__init__()
        self.layers = []

        last_hs = N_ATOM_FEATURES_LP_MODEL

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

        self.layers.append(GCNConv(last_hs, N_ATOM_FEATURES_LP_MODEL))
        self.layers = nn.ModuleList(self.layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(N_ATOM_FEATURES_LP_MODEL * 2, 10), nn.ReLU(), nn.Linear(10, LP_N_CLASSES * 2)
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

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        return self.model(x, edge_index, edge_attr)

    def custom_loss(self, y_hat, y):
        y_hat_1 = y_hat[:, :LP_N_CLASSES]
        y_1 = y[:, 0]

        y_hat_2 = y_hat[:, LP_N_CLASSES:]
        y_2 = y[:, 1]

        loss1 = self.loss(y_hat_1, y_1)
        loss2 = self.loss(y_hat_2, y_2)

        self.log("#lps_loss", loss1.item(), batch_size=self.batch_size)
        self.log("#diff_lps_loss", loss2.item(), batch_size=self.batch_size)

        return loss1 + loss2

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


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric import nn as geom_nn
from torch_geometric.utils import batched_negative_sampling

from .utils import get_matching, convert_batch_to_mask

N_ATOM_FEATURES = 17
N_LP_FEATURES = 19
N_EDGE_FEATURES = 27
N_BOND_FEATURES = 3

GCN_OUTPUT_SIZE = 5

N_ATOM_TARGETS = 4
N_LP_TARGETS_CE = 4
N_LP_TARGETS = 5
N_BOND_TARGETS = 7
LP_OCC_FEATURE_ID = 8
N_INT_TARGETS = 3

FEATURE_NAMES = [
    'atom_'
]


class MLP(nn.Module):
    def __init__(self, input_size, layers, activation='ReLU'):
        super().__init__()

        self.mlp = []
        current_hidden_size = input_size

        for layer in layers:
            self.mlp.append(nn.Linear(current_hidden_size, layer))

            if activation == 'ReLU':
                self.mlp.append(nn.ReLU())
            elif activation == 'Sigmoid':
                self.mlp.append(nn.Sigmoid())
            elif activation == 'Tanh':
                self.mlp.append(nn.Tanh())
            else:
                raise ValueError('Unknown activation')

            current_hidden_size = layer

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class NodeEvolver(nn.Module):
    def __init__(self, embedding_dim, hidden_size, node_target_size, hidden_layers=(256, 128), target_layers=(256, 128),
                 out_transform=()):
        super().__init__()

        self.hidden_transform = MLP(hidden_size, hidden_layers)
        self.target_transform = MLP(node_target_size + embedding_dim, target_layers)

        self.out_transform = MLP(hidden_layers[-1] * 3, list(out_transform) + [hidden_size], 'Tanh')

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding, hidden, node_target, mask):
        hidden_transformed = self.hidden_transform(hidden)  # [N_h, H]
        target_transformed = self.target_transform(torch.hstack((node_target, embedding)))  # [N_t, H]

        weights = self.softmax(
            (hidden_transformed @ target_transformed.T) * mask
        )  # [N_h, N_t] 
        weighted = weights @ target_transformed  # [N_h, H]

        out = hidden + self.out_transform(torch.hstack((weighted, hidden_transformed, target_transformed)))

        return out


class NodeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def log_mse_metric(self, y_hat, y, batch):
        metrics = {}

        y_hat[
        batch.is_lp == 1, N_ATOM_TARGETS: N_ATOM_TARGETS + N_LP_TARGETS_CE
        ] = F.softmax(
            y_hat[batch.is_lp == 1, N_ATOM_TARGETS: N_ATOM_TARGETS + N_LP_TARGETS_CE]
        )

        for i in range(N_ATOM_TARGETS + N_LP_TARGETS + N_BOND_TARGETS):
            if i < N_ATOM_TARGETS:
                slice_ = batch.is_atom == 1
            else:
                if i < N_ATOM_TARGETS + N_LP_TARGETS:
                    slice_ = batch.is_lp == 1
                else:
                    slice_ = batch.is_bond == 1

            metrics[
                f'loss_node_{i}'
            ] = ((y[slice_, i] - y_hat[slice_, i]) ** 2).mean().item()

        return metrics

    def forward(self, y_hat, y, batch):
        loss_atom = self.mse(
            y[batch.is_atom == 1, :N_ATOM_TARGETS],
            y_hat[batch.is_atom == 1, :N_ATOM_TARGETS],
        )

        loss_lp_clf = self.ce(
            y_hat[batch.is_lp == 1, N_ATOM_TARGETS: N_ATOM_TARGETS + N_LP_TARGETS_CE],
            y[batch.is_lp == 1, N_ATOM_TARGETS: N_ATOM_TARGETS + N_LP_TARGETS_CE],
        )

        loss_lp_mse = self.mse(
            y[batch.is_lp == 1, LP_OCC_FEATURE_ID],
            y_hat[batch.is_lp == 1, LP_OCC_FEATURE_ID],
        )

        loss_orbital_mse = self.mse(
            y[batch.is_bond == 1, N_ATOM_TARGETS + N_LP_TARGETS:],
            y_hat[batch.is_bond == 1, N_ATOM_TARGETS + N_LP_TARGETS:],
        )

        loss = loss_atom + loss_lp_clf + loss_lp_mse + loss_orbital_mse

        metrics = {
            'loss_atom': loss_atom.item(),
            'loss_lp_clf': loss_lp_clf.item(),
            'loss_lp_mse': loss_lp_mse.item(),
            'loss_orbital_mse': loss_orbital_mse.item()
        }

        metrics.update(self.log_mse_metric(y_hat, y, batch))

        return loss, metrics


class TotalLoss(nn.Module):
    def __init__(self, perform_matching=True):
        super().__init__()
        self.node_loss = NodeLoss()
        self.link_loss = nn.BCEWithLogitsLoss()
        self.a2b_loss = nn.MSELoss()
        self.int_loss = nn.MSELoss()

        self.perform_matching = perform_matching

    def forward(self, y_hat, y, batch, groups, interaction_edge_index_pos, interaction_edge_index):
        link_preds, a2b_preds, node_preds, int_preds = y_hat
        link_targets, a2b_targets, node_targets, int_targets = y

        # Matching step

        if self.perform_matching:

            matching = get_matching(groups, node_preds, node_targets)
            node_targets = node_targets[matching]

            changed_indexes = []
            for i, match in enumerate(matching):
                if match != i:
                    changed_indexes.append(i)

            changed_indexes = set(changed_indexes)

            # Get columns from interaction edge indexes, which indexes are not changed
            if changed_indexes:
                interaction_edge_index_pos_filtered = sum(
                    interaction_edge_index_pos == idx for idx in changed_indexes).bool()
                interaction_edge_index_pos_filtered = ~(interaction_edge_index_pos_filtered.any(axis=0))

                int_preds = int_preds[interaction_edge_index_pos_filtered]
                int_targets = int_targets[interaction_edge_index_pos_filtered]

                interaction_edge_index_filtered = sum(interaction_edge_index == idx for idx in changed_indexes).bool()
                interaction_edge_index_filtered = ~(interaction_edge_index_filtered.any(axis=0))

                link_preds = link_preds[interaction_edge_index_filtered]
                link_targets = link_targets[interaction_edge_index_filtered]

        # Finally, compute the loss values

        node_loss, metrics = self.node_loss(node_preds, node_targets, batch)
        a2b_loss = self.a2b_loss(a2b_preds, a2b_targets)
        link_loss = self.link_loss(link_preds, link_targets)
        int_loss = self.int_loss(int_preds, int_targets)

        metrics['loss_node'] = node_loss.item()
        metrics['loss_a2b'] = a2b_loss.item()
        metrics['loss_link'] = link_loss.item()
        metrics['loss_int'] = int_loss.item()

        # loss = node_loss
        loss = node_loss + a2b_loss + link_loss + int_loss

        metrics['loss'] = loss.item()

        return loss, metrics


class GNNModel(nn.Module):
    def __init__(self, hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim, gnn_output_dim, heads,
                 use_gnn=True, take_last_only=False, baseline_gnn=None, hidden_dim=256,
                 use_evolver=False, evolver_steps=5):
        super().__init__()
        self.layers = []

        last_hs = N_ATOM_FEATURES + N_LP_FEATURES + N_BOND_FEATURES

        self.use_gnn = use_gnn
        self.take_last_only = take_last_only
        self.baseline_gnn = baseline_gnn

        if use_gnn:
            if baseline_gnn is None:
                print('Using full GNN model')

                all_hss = 0

                for hs in hidden_size:
                    self.layers += [
                        geom_nn.GATConv(last_hs, hs, edge_dim=N_EDGE_FEATURES, heads=heads, concat=False),
                        nn.ReLU(),
                    ]

                    last_hs = hs
                    all_hss += hs

                self.layers.append(geom_nn.GCNConv(last_hs, gnn_output_dim))
                all_hss += gnn_output_dim

                self.layers = nn.ModuleList(self.layers)
            else:
                print('Using baseline GNN model:', baseline_gnn)

                model_dict = {
                    'GAT_tg': geom_nn.GAT,
                    'GCN_tg': geom_nn.GraphSAGE,
                }

                model_params = {
                    'in_channels': N_ATOM_FEATURES + N_LP_FEATURES + N_BOND_FEATURES,
                    'hidden_channels': 1024,
                    'out_channels': gnn_output_dim,
                    'num_layers': 7
                }

                if baseline_gnn == 'GAT_tg':
                    model_params['edge_dim'] = N_EDGE_FEATURES

                self.model = model_dict[baseline_gnn](**model_params)

            if take_last_only:
                print('Using only last GNN layer')
                fcn_input_dim = gnn_output_dim
            else:
                print('Stacking all GNN layer outputs')
                fcn_input_dim = N_ATOM_FEATURES + N_LP_FEATURES + N_BOND_FEATURES + all_hss
        else:
            fcn_input_dim = N_ATOM_FEATURES + N_LP_FEATURES + N_BOND_FEATURES

        self.fcn_head = nn.Sequential(
            nn.Linear(
                fcn_input_dim, fcn_hidden_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(fcn_hidden_dim),
            nn.Linear(fcn_hidden_dim, embedding_dim),
            nn.ReLU()
        )

        if use_evolver:
            print('Using evolver')
            embedding_dim += hidden_dim

        self.link_clf = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + 1, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, 1)
        )

        self.fcn_head_a2b = nn.Sequential(
            nn.Linear(embedding_dim * 2, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, 6)
        )

        self.fcn_head_node = nn.Sequential(
            nn.Linear(embedding_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, N_ATOM_TARGETS + N_LP_TARGETS + N_BOND_TARGETS)
        )

        self.fcn_int_node = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + 1, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, N_INT_TARGETS)
        )

        self.hidden_dim = hidden_dim
        self.evolver = NodeEvolver(embedding_dim, hidden_dim,
                                   N_ATOM_TARGETS + N_LP_TARGETS + N_BOND_TARGETS)

        self.use_evolver = use_evolver
        self.evolver_steps = evolver_steps

    def get_predictions(self, embeddings, interaction_edge_index_pos, interaction_edge_index, xyz_data, vector_data,
                        a2b_index):

        # Link prediction task
        link_preds = embeddings[interaction_edge_index]
        link_preds = torch.cat([link_preds[0], link_preds[1]], axis=1)

        positions = xyz_data[interaction_edge_index]
        vectors = vector_data[interaction_edge_index]

        distances = ((positions[0] - positions[1]) ** 2).sum(axis=1)
        angles = F.cosine_similarity(vectors[0], vectors[1])

        link_preds = torch.hstack([link_preds, distances[:, None], angles[:, None]])
        link_preds = self.link_clf(link_preds)

        # A2b prediction
        a2b_preds = embeddings[a2b_index]
        a2b_preds = torch.cat([a2b_preds[0], a2b_preds[1]], axis=1)
        a2b_preds = self.fcn_head_a2b(a2b_preds)

        # Node prediction
        node_preds = self.fcn_head_node(embeddings)

        # Interaction energy prediction
        int_preds = embeddings[interaction_edge_index_pos]
        int_preds = torch.cat([int_preds[0], int_preds[1]], axis=1)

        positions = xyz_data[interaction_edge_index_pos]
        vectors = vector_data[interaction_edge_index_pos]

        distances = ((positions[0] - positions[1]) ** 2).sum(axis=1)
        angles = F.cosine_similarity(vectors[0], vectors[1])

        int_preds = torch.hstack([int_preds, distances[:, None], angles[:, None]])
        int_preds = self.fcn_int_node(int_preds)

        return link_preds, a2b_preds, node_preds, int_preds

    def forward(self, x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                vector_data, a2b_index, mask):
        if (self.evolver_steps < 1) and self.use_evolver:
            raise ValueError('evolver_steps must be at least 1')

        embeddings = self.get_embedding(x, edge_index, edge_attr)

        intermediate = []
        intermediate_pred = []

        if self.use_evolver:
            hidden_embedding = torch.rand(embeddings.shape[0], self.hidden_dim).to('cuda:0')
            intermediate.append(hidden_embedding.detach().cpu().numpy())

            for evolver_step in range(self.evolver_steps):
                prediction_embedding = torch.hstack([embeddings, hidden_embedding])
                link_preds, a2b_preds, node_preds, int_preds = self.get_predictions(prediction_embedding,
                                                                                    interaction_edge_index_pos,
                                                                                    interaction_edge_index, xyz_data,
                                                                                    vector_data, a2b_index)

                intermediate_pred.append(node_preds.detach().cpu().numpy())

                if evolver_step != self.evolver_steps - 1:  # If not last step
                    hidden_embedding = self.evolver(embeddings, hidden_embedding, node_preds, mask)
                    intermediate.append(hidden_embedding.detach().cpu().numpy())

        else:
            link_preds, a2b_preds, node_preds, int_preds = self.get_predictions(embeddings,
                                                                                interaction_edge_index_pos,
                                                                                interaction_edge_index, xyz_data,
                                                                                vector_data, a2b_index)

        return link_preds, a2b_preds, node_preds, int_preds, intermediate, intermediate_pred

    def get_embedding(self, x, edge_index, edge_attr):
        out = x

        if self.use_gnn:
            if self.baseline_gnn is None:
                each_step = [x]

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

                if self.take_last_only:
                    out = each_step[-1]
                else:
                    out = torch.cat(each_step, dim=1)
            else:
                if self.baseline_gnn in ['GCN_tg']:
                    out = self.model(x, edge_index)
                elif self.baseline_gnn == 'GAT_tg':
                    out = self.model(x, edge_index, edge_attr)
                else:
                    raise ValueError('Unknown model type')

        out = self.fcn_head(out)

        return out


class GNN(pl.LightningModule):
    def __init__(self,
                 hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim, gnn_output_dim, heads,
                 lr=2e-4, use_gnn=True, take_last_only=False, baseline_gnn=None,
                 hidden_dim=256, use_evolver=False, evolver_steps=5, perform_matching=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNModel(hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim,
                              gnn_output_dim, heads, use_gnn, take_last_only, baseline_gnn,
                              hidden_dim, use_evolver, evolver_steps
                              )

        self.loss = TotalLoss(perform_matching)

    def forward(self, x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                vector_data, a2b_index, mask=None):
        if self.hparams.use_evolver:
            assert mask is not None, 'Mask should not be None if use_evolver is True'
        return self.model(x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                          vector_data, a2b_index, mask)

    def get_embedding(self, x, edge_index, edge_attr):
        return self.model.get_embedding(x, edge_index, edge_attr)

    def parse_data(self, data, inplace_sampling=True):
        x, edge_index, edge_attr, interaction_edge_index_pos = data.x, data.edge_index, data.edge_attr, data.interaction_edge_index
        xyz_data, vector_data = data.xyz_data, data.vector_data
        a2b_index, a2b_targets = data.a2b_index, data.a2b_targets
        node_targets = data.y
        int_targets = data.interaction_targets

        target = torch.zeros((interaction_edge_index_pos.shape[1] * 2, 1), device=self.device)
        target[:interaction_edge_index_pos.shape[1]] += 1
        target = target.float()

        if inplace_sampling:
            negative_samples = batched_negative_sampling(interaction_edge_index_pos, data.batch)

        interaction_edge_index = torch.cat([interaction_edge_index_pos, negative_samples], axis=1).long()

        groups = data.groups.cpu().numpy()

        mask = convert_batch_to_mask(data.batch).cuda()

        return (x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index,
                xyz_data, vector_data, a2b_index, mask), (target, a2b_targets, node_targets, int_targets), groups

    def log_scalar_dict(self, metrics, prefix):
        for k, v in metrics.items():
            self.log(
                f'{prefix}/{k}',
                v
            )

    def training_step(self, batch, batch_idx):
        x, (link_targets, a2b_targets, node_targets, int_targets), groups = self.parse_data(batch)

        link_preds, a2b_preds, node_preds, int_preds, _, _ = self.forward(*x)

        loss, metrics = self.loss(
            (link_preds, a2b_preds, node_preds, int_preds),
            (link_targets, a2b_targets, node_targets, int_targets),
            batch, groups, x[3], x[4]
        )

        self.log_scalar_dict(metrics, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        x, (link_targets, a2b_targets, node_targets, int_targets), groups = self.parse_data(batch)

        link_preds, a2b_preds, node_preds, int_preds, _, _ = self.forward(*x)

        loss, metrics = self.loss(
            (link_preds, a2b_preds, node_preds, int_preds),
            (link_targets, a2b_targets, node_targets, int_targets),
            batch, groups, x[3], x[4]
        )

        self.log_scalar_dict(metrics, 'val')

        return loss

    def test_step(self, batch, batch_idx):
        x, (link_targets, a2b_targets, node_targets, int_targets), groups = self.parse_data(batch)

        link_preds, a2b_preds, node_preds, int_preds, _, _ = self.forward(*x)

        loss, metrics = self.loss(
            (link_preds, a2b_preds, node_preds, int_preds),
            (link_targets, a2b_targets, node_targets, int_targets),
            batch, groups, x[3], x[4]
        )

        self.log_scalar_dict(metrics, 'test')

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
