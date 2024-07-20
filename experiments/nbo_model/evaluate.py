import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

from model import GNN


def make_preds(data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    xyz_data, vector_data = data.xyz_data, data.vector_data
    a2b_index, a2b_targets = data.a2b_index, data.a2b_targets

    interaction_edge_index = [[i, j] for i in range(len(x)) for j in range(len(x))]
    interaction_edge_index = torch.LongTensor(interaction_edge_index).T

    preds, a2b_preds, node_preds, int_preds = gnn.forward(
        x, edge_index, edge_attr, data.interaction_edge_index, interaction_edge_index,
        xyz_data, vector_data, a2b_index
    )

    preds = torch.sigmoid(preds.reshape((len(x), len(x)))).cpu().detach().numpy()

    gt = np.zeros((len(x), len(x)))
    for i, j in data.interaction_edge_index.long().T:
        gt[i, j] = 1

    n_atoms = int(data.is_atom.sum())
    preds = preds[n_atoms:, n_atoms:]
    np.fill_diagonal(preds, 0)

    gt = gt[n_atoms:, n_atoms:]

    symbol = data.symbol[n_atoms:]
    index = np.arange(len(x))[n_atoms:]

    roc_auc = roc_auc_score(gt.reshape(-1), preds.reshape(-1))

    return (preds, gt, symbol, index, roc_auc), (a2b_preds, node_preds, int_preds), (
        a2b_targets, data.y, data.interaction_targets)


def make_preds_no_gt(data, gnn):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    xyz_data, vector_data = data.xyz_data, data.vector_data
    a2b_index = data.a2b_index

    interaction_edge_index = [[i, j] for i in range(len(x)) for j in range(len(x))]
    interaction_edge_index = torch.LongTensor(interaction_edge_index).T

    preds, a2b_preds, node_preds, int_preds = gnn.forward(
        x, edge_index, edge_attr, data.interaction_edge_index, interaction_edge_index,
        xyz_data, vector_data, a2b_index
    )

    preds = torch.sigmoid(preds.reshape((len(x), len(x)))).cpu().detach().numpy()

    n_atoms = int(data.is_atom.sum())
    preds = preds[n_atoms:, n_atoms:]
    np.fill_diagonal(preds, 0)

    symbol = data.symbol[n_atoms:]
    index = np.arange(len(x))[n_atoms:]

    return (preds, symbol, index), (a2b_preds, node_preds, int_preds)


def compute_metrics(true, pred):
    metrics = f"{np.sqrt(mean_squared_error(true, pred)):.3f} {mean_absolute_error(true, pred):.3f} {r2_score(true, pred):.3f}"
    print(' & '.join(metrics.split(' ')))

    return metrics


def get_data(graphs_path):
    if os.path.isfile(graphs_path):
        data = torch.load(graphs_path)
    else:
        data = sum([
            torch.load(os.path.join(graphs_path, file)) for file in tqdm(os.listdir(graphs_path)) if
            file.endswith('.pt')
        ], [])

    print("Initial data size:", len(data))

    nice_graphs = []

    for graph in tqdm(data):
        conditions = [
            graph.x.shape[1] != 17,
            graph.y.shape[1] != 16,
            graph.edge_attr.shape[1] != 16,
        ]

        if any(conditions):
            print("Skipping graph with wrong shape:", graph.x.shape, graph.y.shape, graph.edge_attr.shape)
            continue

        nice_graphs.append(
            Data(
                x=torch.FloatTensor(graph.x),
                y=torch.FloatTensor(graph.y),
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
                symbol=graph.symbol
            )
        )

    del data
    test = [graph for graph in nice_graphs if graph.type == 'test']

    # Print dataset sizes
    print(f'Test: {len(test)}')
    return test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--model_path", type=str, help="Path to the model")

    args = parser.parse_args()

    out_dir = args.model_path + '_results'
    if os.path.exists(out_dir):
        raise ValueError(f"{out_dir} already exists")

    os.makedirs(out_dir)

    test = get_data(args.graphs_path)

    gnn = GNN.load_from_checkpoint(args.model_path)
    gnn.eval()

    roc_aucs = []
    y_hat = []
    y = []

    out = []
    out_a2b = []
    out_int = []

    with torch.no_grad():
        for mol in tqdm(test):
            preds = make_preds(mol)

            out.append(preds[1][1])
            out_a2b.append(preds[1][0])
            out_int.append(preds[1][2])

            y_hat.append(preds[0][0])
            y.append(preds[0][1])
            roc_aucs.append(preds[0][-1])

    compute_metrics(
        np.concatenate(y, axis=None),
        np.concatenate(y_hat, axis=None)
    )

    print("ROC AUC score:", roc_auc_score(
        np.concatenate(y, axis=None),
        np.concatenate(y_hat, axis=None)
    ))

    print("Average ROC AUC:", np.mean(roc_aucs))

    out_data = torch.vstack(out).detach().numpy()
    out_a2b_data = torch.vstack(out_a2b).detach().numpy()
    out_int = torch.vstack(out_int).detach().numpy()

    y_true = np.vstack([mol.y for mol in test])
    y_a2b_true = np.vstack([mol.a2b_targets for mol in test])
    y_int_true = np.vstack([mol.interaction_targets for mol in test])

    is_atom = np.hstack([mol.is_atom for mol in test])
    is_lp = np.hstack([mol.is_lp for mol in test])
    is_bond = np.hstack([mol.is_bond for mol in test])

    symbol = np.hstack([mol.symbol for mol in test])

    # Save all output data as numpy arrays in the numpy folder
    os.makedirs(os.path.join(out_dir, 'numpy'))
    np.save(os.path.join(out_dir, 'numpy', 'out.npy'), out_data)
    np.save(os.path.join(out_dir, 'numpy', 'out_a2b.npy'), out_a2b_data)
    np.save(os.path.join(out_dir, 'numpy', 'out_int.npy'), out_int)
    np.save(os.path.join(out_dir, 'numpy', 'y.npy'), np.concatenate(y, axis=None))
    np.save(os.path.join(out_dir, 'numpy', 'y_hat.npy'), np.concatenate(y_hat, axis=None))
    np.save(os.path.join(out_dir, 'numpy', 'roc_aucs.npy'), roc_aucs)
    np.save(os.path.join(out_dir, 'numpy', 'y_true.npy'), y_true)
    np.save(os.path.join(out_dir, 'numpy', 'y_a2b_true.npy'), y_a2b_true)
    np.save(os.path.join(out_dir, 'numpy', 'y_int_true.npy'), y_int_true)
    np.save(os.path.join(out_dir, 'numpy', 'is_atom.npy'), is_atom)
    np.save(os.path.join(out_dir, 'numpy', 'is_lp.npy'), is_lp)
    np.save(os.path.join(out_dir, 'numpy', 'is_bond.npy'), is_bond)
    np.save(os.path.join(out_dir, 'numpy', 'symbol.npy'), symbol)

    # Atom predictions

    y_mean = np.array([1.6833026e-05, 9.7693789e-01, 2.6718247e+00, 3.6606803e+00])
    y_std = np.array([0.34014496, 0.9994328, 2.0642664, 3.03501])

    out_data_atom = out_data.copy()[:, :4]
    out_data_atom *= y_std
    out_data_atom += y_mean

    out_data_atom = out_data_atom[is_atom == 1]

    y_true_atom = y_true.copy()[:, :4]
    y_true_atom = y_true_atom[is_atom == 1]

    # Lone pair predictions

    y_mean = np.array([0, 0, 0, 0, 1.9150e+00])
    y_std = np.array([1, 1, 1, 1, 8.9718e-02])

    out_data_lp = out_data.copy()[:, 4:9]
    out_data_lp *= y_std
    out_data_lp += y_mean

    out_data_lp = out_data_lp[is_lp == 1]
    out_data_lp[:, :4] = (np.exp(out_data_lp[:, :4]) / np.sum(np.exp(out_data_lp[:, :4]), axis=1)[:, None])

    y_true_lp = y_true.copy()[:, 4:9]
    y_true_lp = y_true_lp[is_lp == 1]
    y_true_lp[:, :4] = y_true_lp[:, :4] / 100

    # Bond predictions
    out_data_bond = out_data.copy()[:, 9:]
    out_data_bond = out_data_bond[is_bond == 1]

    y_true_bond = y_true.copy()[:, 9:]
    y_true_bond = y_true_bond[is_bond == 1]

    sample = np.random.choice(len(y_true_atom), int(len(y_true_atom) * 0.05))

    print('Atom')
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate([0, 1, 2, 3]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_true_atom[sample, idx], out_data_atom[sample, idx], s=2)
        plt.plot(
            [y_true_atom[sample, idx].min(), y_true_atom[sample, idx].max()],
            [y_true_atom[sample, idx].min(), y_true_atom[sample, idx].max()], c='gray')

        plt.title(compute_metrics(y_true_atom[:, idx], out_data_atom[:, idx]))

    plt.savefig(f'{out_dir}/atom_predictions.png', dpi=300, bbox_inches='tight')

    sample = np.random.choice(len(y_true_lp), int(len(y_true_lp) * 0.05))

    print('Lone pair')
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate([0, 1, 2, 3, 4]):
        plt.subplot(2, 3, i + 1)
        plt.scatter(y_true_lp[sample, idx], out_data_lp[sample, idx], s=2)
        plt.plot(
            [y_true_lp[sample, idx].min(), y_true_lp[sample, idx].max()],
            [y_true_lp[sample, idx].min(), y_true_lp[sample, idx].max()], c='gray')

        plt.title(compute_metrics(y_true_lp[:, idx], out_data_lp[:, idx]))

    plt.savefig(f'{out_dir}/lp_predictions.png', dpi=300, bbox_inches='tight')

    sample = np.random.choice(len(y_true_bond), int(len(y_true_bond) * 0.05))

    print('Bond')
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6]):
        plt.subplot(3, 3, i + 1)
        plt.scatter(y_true_bond[sample, idx], out_data_bond[sample, idx], s=1)
        plt.plot(
            [y_true_bond[sample, idx].min(), y_true_bond[sample, idx].max()],
            [y_true_bond[sample, idx].min(), y_true_bond[sample, idx].max()], c='gray')

        plt.title(compute_metrics(y_true_bond[:, idx], out_data_bond[:, idx]))

    plt.savefig(f'{out_dir}/bond_predictions.png', dpi=300, bbox_inches='tight')

    sample = np.random.choice(len(y_a2b_true), int(len(y_a2b_true) * 0.05))

    print('A2B')
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
        plt.subplot(2, 3, i + 1)
        plt.scatter(y_a2b_true[sample, idx], out_a2b_data[sample, idx], s=1)
        plt.plot(
            [y_a2b_true[sample, idx].min(), y_a2b_true[sample, idx].max()],
            [y_a2b_true[sample, idx].min(), y_a2b_true[sample, idx].max()], c='gray')

        plt.title(compute_metrics(y_a2b_true[:, idx], out_a2b_data[:, idx]))

    plt.savefig(f'{out_dir}/a2b_predictions.png', dpi=300, bbox_inches='tight')

    sample = np.random.choice(len(y_int_true), int(len(y_int_true) * 0.05))

    print('Int')
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate([0, 1, 2]):
        plt.subplot(2, 3, i + 1)

        plt.scatter(y_int_true[sample, idx], out_int[sample, idx], s=1)
        plt.plot(
            [y_int_true[sample, idx].min(), y_int_true[sample, idx].max()],
            [y_int_true[sample, idx].min(), y_int_true[sample, idx].max()], c='gray')

        plt.title(compute_metrics(y_int_true[:, idx], out_int[:, idx]))

    plt.savefig(f'{out_dir}/int_predictions.png', dpi=300, bbox_inches='tight')
