import os

import yaml
import logging
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.autonotebook import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from model import GNN

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    parser.add_argument("--training_fraction", type=float, default=1.0)
    parser.add_argument("--attempt", type=int, default=0)

    args = parser.parse_args()

    BS = 1024

    file2y = np.loadtxt(f'/home/dboiko/SPAHM/SPAHM/1_QM7/target/{args.target}.dat')
    test_idx = set(list(np.loadtxt('/home/dboiko/SPAHM/test-indices/QM7_test_indices.dat').astype(int)))

    data = torch.load('/home/dboiko/reproducing_experiments/qm7_graphs_from_repo.pt')
    nice_graphs = []

    logging.info('Clean up')
    for graph in tqdm(data):
        nice_graphs.append(
            Data(
                x=torch.FloatTensor(graph.x),
                y=torch.FloatTensor([file2y[int(graph.qm9_id.split('_')[1])]]),
                edge_index=torch.LongTensor(graph.edge_index),
                edge_attr=torch.FloatTensor(graph.edge_attr),
                type='train' if int(graph.qm9_id.split('_')[1]) not in test_idx else 'test'
            )
        )

    train = [graph for graph in nice_graphs if graph.type == 'train']

    train_idx, val_idx = train_test_split(range(len(train)), test_size=0.1, random_state=42)
    train_idx = np.random.choice(train_idx, int(len(train_idx) * args.training_fraction), replace=False)

    val = [train[i] for i in val_idx]
    train = [train[i] for i in train_idx]

    test = [graph for graph in nice_graphs if graph.type == 'test']

    y_mean = np.mean([graph.y.numpy() for graph in train], axis=0)
    y_std = np.std([graph.y.numpy() for graph in train], axis=0)

    os.makedirs(f'./logs_long/{args.target}_{args.training_fraction}_{args.attempt}', exist_ok=True)
    np.save(f'./logs_long/{args.target}_{args.training_fraction}_{args.attempt}/mean_std.npy', [y_mean, y_std])

    for split in [train, val, test]:
        for graph in split:
            graph.y -= y_mean
            graph.y /= y_std

    print(len(train), len(val), len(test))

    train_loader = DataLoader(train, batch_size=min(
        len(train), BS
    ), shuffle=True, drop_last=True, num_workers=3)
    val_loader = DataLoader(val, batch_size=BS, num_workers=3)
    test_loader = DataLoader(test, batch_size=BS, num_workers=3)

    with open('model_config_GCN_tg.yaml', "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

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
    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback_train,
                                    checkpoint_callback_val,
                                    checkpoint_callBack_last,
                                    checkpoint_callback_every_n_epochs],
                         check_val_every_n_epoch=10, max_epochs=5000, gradient_clip_val=1.0,
                         default_root_dir=f'./logs_long/{args.target}_{args.training_fraction}_{args.attempt}')

    trainer.fit(gnn, train_loader, val_loader)
    trainer.test(gnn, test_loader)
