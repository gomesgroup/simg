import argparse
import yaml
import os
import random
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.autonotebook import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from model import GNN

from nbo_gcnn.graph_construction import convert_NBO_graph_to_downstream

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
parser.add_argument(
    "--splits_path",
    type=str,
    help="Path to the splits. Must contain {train,val,test}.txt files.",
)
parser.add_argument("--bs", type=int, help="Batch size")
parser.add_argument("--model_config", type=str, help="Path to the config of the model")
parser.add_argument("--sample", type=float, default=1.0, help='Do sampling?')
parser.add_argument("--parts", action="store_true", help='In parts?')
parser.add_argument("--add_s", type=str, default="", help='Suffix to add to the name')
parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

# Check inputs
if hparams.parts:
    assert os.path.exists(hparams.graphs_path), "Graphs path does not exist"
assert os.path.exists(hparams.model_config), "Model config does not exist"

logging.info(f'Loading graphs from {hparams.graphs_path}')
if not hparams.parts:
    data = torch.load(hparams.graphs_path)
else:
    data = sum([
        torch.load(os.path.join(hparams.graphs_path, file)) for file in tqdm(os.listdir(hparams.graphs_path)) if
        file.endswith('.pt')
    ], [])

data = [mol for mol in data if mol.type != 'remove']

if hparams.from_NBO:
    logging.info('Converting to downstream format')
    data = [convert_NBO_graph_to_downstream(graph, molecular_only=hparams.molecular) for graph in tqdm(data)]

for mol in data:
    mol.y = [mol.normalized_targets[i][0] for i in [0, 1, 2, 3, 4, 5, 6, 11]]

CHEMICAL_ACC_NORMALISING_FACTORS = [0.066513725, 0.012235489, 0.071939046,
                                    0.033730778, 0.033486113, 0.004278493,
                                    0.001330901, 0.004165489, 0.004128926,
                                    0.00409976, 0.004527465, 0.012292586,
                                    0.037467458]

recalc_mae = [CHEMICAL_ACC_NORMALISING_FACTORS[i] for i in [0, 1, 2, 3, 4, 5, 6, 11]]

nice_graphs = []

logging.info('Clean up')
for graph in tqdm(data):
    nice_graphs.append(
        Data(
            x=torch.FloatTensor(graph.x[:, :17]),
            y=torch.FloatTensor([graph.y]),
            edge_index=torch.LongTensor(graph.edge_index),
            edge_attr=torch.FloatTensor(graph.edge_attr),
            type=graph.type
        )
    )

del data

train = [graph for graph in nice_graphs if graph.type == 'train']
val = [graph for graph in nice_graphs if graph.type == 'valid']
test = [graph for graph in nice_graphs if graph.type == 'test']

print(len(train), len(val), len(test))

if hparams.sample != 1.0:
    print(f"Reducing from {len(train)}")
    train = random.sample(train, int(len(train) * hparams.sample))
    print(f"\tto {len(train)}")

train_loader = DataLoader(train, batch_size=hparams.bs, shuffle=True, drop_last=True, num_workers=12)
val_loader = DataLoader(val, batch_size=hparams.bs, num_workers=12)
test_loader = DataLoader(test, batch_size=hparams.bs, num_workers=12)

with open(hparams.model_config, "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

if (not model_config['model_type'].endswith('_tg')) and model_config['model_type'] != 'FiLM':
    model_config['model_params']['input_features'] = nice_graphs[0].x.shape[1]
    model_config['model_params']['out_targets'] = len(nice_graphs[0].y.shape[1])
    model_config['model_params']['edge_features'] = nice_graphs[0].edge_attr.shape[1]
else:
    model_config['model_params']['in_channels'] = nice_graphs[0].x.shape[1]

    if model_config['model_type'] == 'GAT_tg':
        model_config['model_params']['edge_dim'] = nice_graphs[0].edge_attr.shape[1]

model_config['recalc_mae'] = recalc_mae

checkpoint_callback_val = ModelCheckpoint(
    monitor='val_loss',
    filename='best_val_{epoch:02d}-{val_loss:.4f}-{train_loss:.4f}',
    save_top_k=5,
    mode='min',
)

checkpoint_callback_train = ModelCheckpoint(
    monitor='train_loss',
    filename='best_train_{epoch:02d}-{val_loss:.4f}-{train_loss:.4f}',
    save_top_k=5,
    mode='min',
)

checkpoint_callBack_last = ModelCheckpoint(filename='last')

checkpoint_callback_every_n_epochs = ModelCheckpoint(every_n_epochs=50)

gnn = GNN(**model_config)
trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[checkpoint_callback_train,
                                                            checkpoint_callback_val,
                                                            checkpoint_callBack_last,
                                                            checkpoint_callback_every_n_epochs])
trainer.fit(gnn, train_loader, val_loader)

trainer.test(gnn, test_loader)
