import argparse
import os
import random
import pickle as pkl

import torch
import pytorch_lightning as pl
from tqdm.autonotebook import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from data import split
from model import GNN

parser = argparse.ArgumentParser()
parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
parser.add_argument("--normalizer_path", type=str, help="Path to the normalizers")
parser.add_argument(
    "--splits_path",
    type=str,
    help="Path to the splits. Must contain {train,val,test}.txt files.",
)
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument("--sample", type=float, default=1.0, help='Do sampling?')
parser.add_argument("--column", type=int, default=-1, help='Column to use')
parser.add_argument("--parts", type=int, default=1, help='Number of parts')
parser.add_argument("--add_s", type=str, default="", help='Suffix to add to the name')
parser.add_argument("--reproduce", type=str, default="", help='Reproduce another paper')

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

# Check inputs
if hparams.parts == 1:
    assert os.path.exists(hparams.graphs_path), "Graphs path does not exist"
assert os.path.exists(hparams.normalizer_path), "Normalizer path does not exist"
assert os.path.exists(hparams.splits_path), "Splits path does not exist"

if hparams.parts == 1:
    data = torch.load(hparams.graphs_path)
else:
    data = sum(
        [torch.load(f'{hparams.graphs_path}_{i}' + ('' if not hparams.add_s else '_' + hparams.add_s) + '.pt') for i
         in range(hparams.parts)], [])

with open(hparams.normalizer_path, "rb") as f:
    normalizers = pkl.load(f)

columns = [
    'mu', 'alpha',
    'homo', 'lumo', 'gap',
    'r2', 'zpve',
    'cv'
]

if hparams.column != -1:
    columns = [columns[hparams.column]]

if not hparams.reproduce:
    for mol in data:
        mol.y = [(mol.y[column] - normalizers[0][column]) / normalizers[1][column] for column in columns]
elif hparams.reproduce == 'full':
    print('Using reproduction mode: full')

    data = [mol for mol in data if mol.type != 'remove']

    for mol in data:
        mol.y = [mol.normalized_targets[i][0] for i in [0, 1, 2, 3, 4, 5, 6, 11]]

    CHEMICAL_ACC_NORMALISING_FACTORS = [0.066513725, 0.012235489, 0.071939046,
                                        0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926,
                                        0.00409976, 0.004527465, 0.012292586,
                                        0.037467458]

    recalc_mae = [CHEMICAL_ACC_NORMALISING_FACTORS[i] for i in [0, 1, 2, 3, 4, 5, 6, 11]]
else:
    raise ValueError('Unknown reproduce option')

nice_graphs = []

for graph in tqdm(data):
    nice_graphs.append(
        Data(
            x=torch.FloatTensor(graph.x),
            y=torch.FloatTensor([graph.y]),
            edge_index=torch.LongTensor(graph.edge_index).T,
            edge_attr=torch.FloatTensor(graph.edge_attr),
            type=graph.type
        )
    )

del data

if not hparams.reproduce:
    train, val, test = split(nice_graphs, hparams.splits_path)
else:
    train = [graph for graph in nice_graphs if graph.type == 'train']
    val = [graph for graph in nice_graphs if graph.type == 'valid']
    test = [graph for graph in nice_graphs if graph.type == 'test']

print(len(train), len(val), len(test))

if hparams.sample != 1.0:
    print(f"Reducing from {len(train)}")
    train = random.sample(train, int(len(train) * hparams.sample))
    print(f"\tto {len(train)}")

train_loader = DataLoader(train, batch_size=hparams.bs, shuffle=True, num_workers=6)
val_loader = DataLoader(val, batch_size=hparams.bs, num_workers=6)
test_loader = DataLoader(test, batch_size=hparams.bs, num_workers=6)

for root, dir, files in os.walk('.'):
    for file in files:
        if not file.endswith('.ckpt'):
            continue

        print('Starting processing the file:', file)

        gnn = GNN.load_from_checkpoint(checkpoint_path=os.path.join(root, file))
        trainer = pl.Trainer.from_argparse_args(hparams)

        trainer.test(gnn, test_loader)
