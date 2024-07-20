import os
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data

from evaluate import make_preds_no_gt
from model import GNN

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate graphs for downstream tasks")
    parser.add_argument("--graphs_path", type=str, required=True, help="Path to the graphs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output")

    hparams = parser.parse_args()

    if os.path.isfile(hparams.graphs_path):
        data = torch.load(hparams.graphs_path)
    else:
        data = sum([
            torch.load(os.path.join(hparams.graphs_path, file)) for file in tqdm(os.listdir(hparams.graphs_path)) if
            file.endswith('.pt')
        ], [])

    print("Data size:", len(data))

    nice_graphs = []

    for graph in tqdm(data):
        if graph.type == 'remove':
            continue

        conditions = [
            graph.x.shape[1] != 17,
            graph.y.shape[1] != 16,
            graph.edge_attr.shape[1] != 16,
        ]

        if any(conditions):
            print("Skipping graph with wrong shape:", graph.x.shape, graph.y.shape, graph.edge_attr.shape)
            continue

        y_mean = np.array([1.6833026e-05, 9.7693789e-01, 2.6718247e00, 3.6606803e00, 0, 0, 0, 0, 1.9150e00, ] + [0] * 7)
        y_std = np.array([0.34014496, 0.9994328, 2.0642664, 3.03501, 100, 100, 100, 100, 8.9718e-02, ] + [1] * 7)

        y = graph.y.numpy()

        y -= y_mean
        y /= y_std

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
                symbol=graph.symbol,
                normalized_targets=graph.normalized_targets
            )
        )

    print("Nice graphs size:", len(nice_graphs))

    model = GNN.load_from_checkpoint(hparams.model_path)
    model.eval()

    for i, graph in enumerate(tqdm(nice_graphs)):
        try:
            with torch.no_grad():
                (preds, symbol, index), (a2b_preds, node_preds, int_preds) = make_preds_no_gt(graph, model)

            # Check shapes in case of wrong modle
            assert node_preds.shape == graph.y.shape
            assert a2b_preds.shape == graph.a2b_targets.shape
            assert int_preds.shape == graph.interaction_targets.shape

            graph.y = node_preds
            graph.a2b_targets = a2b_preds
            graph.interaction_targets = int_preds
        except IndexError:
            print('IndexError for graph', i)
            graph.type = 'remove'

    # Save all graphs
    torch.save(nice_graphs, hparams.output_path)
