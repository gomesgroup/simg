import argparse
import yaml
import os
import random

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import GNN
from utils import get_group_idx

parser = argparse.ArgumentParser()
parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
parser.add_argument("--bs", type=int, help="Batch size")
parser.add_argument("--model_config", type=str, help="Path to the config of the model")
parser.add_argument("--sample", type=float, default=1.0, help='Do sampling?')

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

if os.path.isfile(hparams.graphs_path):
    data = torch.load(hparams.graphs_path)
else:
    data = sum([
        torch.load(os.path.join(hparams.graphs_path, file)) for file in tqdm(os.listdir(hparams.graphs_path)) if
        file.endswith('_merged.pt')
    ], [])

print("Initial data size:", len(data))

nice_graphs = []

for graph_id, graph in enumerate(tqdm(data)):
    conditions = [
        graph.x.shape[1] != 39,
        graph.y.shape[1] != 16,
        graph.edge_attr.shape[1] != 27,
    ]

    if any(conditions):
        print("Skipping graph with wrong shape:", graph.x.shape, graph.y.shape, graph.edge_attr.shape)
        continue

    y_mean = np.array([1.6833026e-05, 9.7693789e-01, 2.6718247e00, 3.6606803e00, 0, 0, 0, 0, 1.9150e00, ] + [0] * 7)
    y_std = np.array([0.34014496, 0.9994328, 2.0642664, 3.03501, 100, 100, 100, 100, 8.9718e-02, ] + [1] * 7)

    y = graph.y.numpy()

    y -= y_mean
    y /= y_std

    group_idx = get_group_idx(graph)
    group_idx[group_idx != -1] += graph_id * 1000

    nice_graphs.append(
        Data(
            x=torch.FloatTensor(graph.x),
            y=torch.FloatTensor(y),
            edge_index=torch.LongTensor(graph.edge_index),
            edge_attr=torch.FloatTensor(graph.edge_attr),
            is_atom=torch.Tensor(graph.is_atom),
            is_lp=torch.Tensor(graph.is_lp),
            is_bond=torch.Tensor(graph.is_bond),
            interaction_edge_index=torch.LongTensor(graph.interaction_edge_index),
            xyz_data=torch.FloatTensor(graph.xyz_data),
            vector_data=torch.FloatTensor(graph.vector_data),
            a2b_index=torch.LongTensor(graph.a2b_index),
            a2b_targets=torch.FloatTensor(graph.a2b_targets),
            interaction_targets=torch.FloatTensor(graph.interaction_targets),
            type=graph.type,
            groups=torch.LongTensor(group_idx)
        )
    )

del data

train = [graph for graph in nice_graphs if graph.type == 'train']
val = [graph for graph in nice_graphs if graph.type in ['valid', 'val']]
test = [graph for graph in nice_graphs if graph.type == 'test']

# Print dataset sizes
print(f'Train: {len(train)}')
print(f'Val: {len(val)}')
print(f'Test: {len(test)}')

print("Final data size:", len(train) + len(val) + len(test))

if hparams.sample != 1.0:
    print(f"Reducing from {len(train)}")
    train = random.sample(train, int(len(train) * hparams.sample))
    print(f"\tto {len(train)}")

train_loader = DataLoader(train, batch_size=hparams.bs, shuffle=True, num_workers=6, drop_last=True)
val_loader = DataLoader(val, batch_size=hparams.bs, shuffle=False, num_workers=6)
test_loader = DataLoader(test, batch_size=hparams.bs, shuffle=False, num_workers=6)

with open(hparams.model_config, "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

gnn = GNN(**model_config)
print(gnn.model)

checkpoint_callback_val = ModelCheckpoint(
    monitor='val/loss',
    filename='best_val_{epoch:02d}-{val/loss:.4f}-{train/loss:.4f}',
    save_top_k=5,
    mode='min',
)

checkpoint_callback_train = ModelCheckpoint(
    monitor='train/loss',
    filename='best_train_{epoch:02d}-{val/loss:.4f}-{train/loss:.4f}',
    save_top_k=5,
    mode='min',
)

checkpoint_callBack_last = ModelCheckpoint(filename='last')
checkpoint_callback_every_n_epochs = ModelCheckpoint(every_n_epochs=50)

trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[checkpoint_callback_train,
                                                            checkpoint_callback_val,
                                                            checkpoint_callBack_last,
                                                            checkpoint_callback_every_n_epochs])
trainer.fit(gnn, train_loader, val_loader)
