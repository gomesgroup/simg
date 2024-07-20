import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import GNN_LP

import argparse
import yaml
import os

MAX_EPOCHS = 300
GRADIENT_CLIP = 0.0005

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--model_config", type=str, help="Path to the config of the model")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--parts", type=int, default=1, help="Number of parts which the dataset is divided")
    args = parser.parse_args()

    config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    if args.parts == 1:
        assert os.path.exists(args.graphs_path), "Graphs path does not exist"
    assert os.path.exists(args.model_config), "Model config does not exist"

    if args.parts == 1:
        data = torch.load(args.graphs_path)
    else:
        data = sum(
            [torch.load(f"{args.graphs_path}_{i}_merged.pt") for i in range(args.parts)],
            [],
        )

    train = [x for x in data if x.type == "train"]
    val = [x for x in data if x.type == "val"]

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    train_loader = DataLoader(train, batch_size=config["batch_size"], num_workers=4)
    val_loader = DataLoader(val, batch_size=config["batch_size"], num_workers=4)

    model = GNN_LP(config, deg)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=args.gpus,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", min_delta=0, patience=25, mode="min"  # , verbose=True
            )
        ],
        gradient_clip_val=GRADIENT_CLIP,
    )

    trainer.fit(model, train_loader, val_loader)
