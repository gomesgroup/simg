import os
import uuid
import io
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

import redis
from rq import Worker, Queue, Connection, get_current_job

listen = ['default']
redis_url = os.environ.get('REDIS_HOST')
conn = redis.from_url(redis_url)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax(x):
    return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)


def transform_edge_predictions(predictions, n_total, n_start):
    n_nodes = n_total - n_start
    n_features = predictions.shape[1]

    result = np.full((n_nodes * n_nodes, n_features), np.nan)

    # Create edge index mapping
    edge_idx = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                # Convert 2D indices to flattened 1D index
                flat_idx = i * n_nodes + j
                result[flat_idx] = predictions[edge_idx]
                edge_idx += 1

    return result


def plot_homo_molecule(graph):
    from simg.config import COLOR_MAPPER
    # returns bytes

    multiplier = 1 if len(graph.symbol) < 200 else 2
    f, (ax) = plt.subplots(1, 1, figsize=(10*multiplier, 7*multiplier))

    G = nx.Graph()
    G.add_nodes_from([
        (i, {"color": COLOR_MAPPER[symbol]}) for i, symbol in enumerate(graph['symbol'])
    ])

    colors = [G.nodes[n]['color'] for n in G.nodes()]

    if type(graph.edge_index) is not list:
        edges = [tuple(t) for t in graph.edge_index.T.tolist()]
    else:
        edges = graph.edge_index

    G.add_edges_from(edges)
    nx.draw_kamada_kawai(G, node_color=colors, with_labels=True, ax=ax)

    # Add legend
    unique_symbols = sorted(set(graph['symbol']), key=lambda x: list(COLOR_MAPPER.keys()).index(x))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAPPER[symbol], markersize=10, label=symbol) for symbol in unique_symbols]
    ax.legend(handles=handles)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(f)
    return buf.getvalue()


def plot_homo_molecule_targets(graph, targets):
    from simg.config import COLOR_MAPPER
    # returns bytes

    targets = np.array(targets)[:, 0]
    max_abs = np.max(np.abs(targets))
    targets /= max_abs*2
    targets += 0.5

    multiplier = 1 if len(graph.symbol) < 200 else 2
    f, (ax) = plt.subplots(1, 1, figsize=(10*multiplier, 7*multiplier))

    G = nx.Graph()
    G.add_nodes_from([
        (i, {"color": tuple(plt.cm.bwr_r(targets[i])[:3]) if symbol not in ['BND', 'LP'] else COLOR_MAPPER[symbol]}) for i, symbol in enumerate(graph['symbol'])
    ])

    colors = [G.nodes[n]['color'] for n in G.nodes()]

    if type(graph.edge_index) is not list:
        edges = [tuple(t) for t in graph.edge_index.T.tolist()]
    else:
        edges = graph.edge_index

    G.add_edges_from(edges)
    nx.draw_kamada_kawai(G, node_color=colors, with_labels=True, ax=ax)

    cax = plt.axes([0.85, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr_r, norm=plt.Normalize(vmin=-max_abs, vmax=max_abs))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Atomic Charge')

    # Add legend for BND and LP
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAPPER['BND'], markersize=10, label='BND'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAPPER['LP'], markersize=10, label='LP')
    ]
    ax.legend(handles=handles)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(f)
    return buf.getvalue()


def plot_interaction_matrix(preds, symbol, index):
    f, (ax) = plt.subplots(1, 1, figsize=(15, 15))

    lp_lp = deepcopy(preds)
    bnd_lp = deepcopy(preds)
    lp_bnd = deepcopy(preds)
    bnd_bnd = deepcopy(preds)

    mat = np.array(
        [[f"{a}->{b}" for b in symbol] for a in symbol]
    )

    # set all values that are not LP to None
    lp_lp[mat != "LP->LP"] = None
    bnd_lp[mat != "LP->BND"] = None
    lp_bnd[mat != "BND->LP"] = None
    bnd_bnd[mat != "BND->BND"] = None

    extent = [
        index[0]-0.5,
        index[-1]+0.5,
        index[-1]+0.5,
        index[0]-0.5,
    ]
    ax.imshow(lp_lp, cmap="Reds", extent=extent)
    ax.imshow(bnd_lp, cmap="Purples", extent=extent)
    ax.imshow(lp_bnd, cmap="Purples", extent=extent)
    ax.imshow(bnd_bnd, cmap="Blues", extent=extent)
    # ticks
    # array of x-axis ticks method.index_1 from 10 to 10
    index_new = np.arange(
        index[0], index[-1], int(len(index) // 6)
    )
    ax.set_xticks(index_new, fontsize=1)
    ax.set_yticks(index_new, fontsize=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(f)

    return buf.getvalue()


def submit_request(param):
    from simg.model_utils import pipeline
    from simg.data import get_connectivity_info

    job = get_current_job(conn)

    job.meta['status'] = 'Running'
    job.save_meta()

    if '\n' not in param:
        path = uuid.uuid4().hex
        xyz_path = f"{path}.xyz"
        smi_path = f"{path}.smi"

        with open(smi_path, "w") as f:
            f.write(param + '\n')

        os.system(f'obabel -i smi {smi_path} -o xyz -O {xyz_path} --gen3d >/dev/null 2>&1')

        with open(xyz_path, "r") as f:
            xyz = f.read()

        job.meta['status'] = 'Convered to xyz'
        job.save_meta()

        os.remove(xyz_path)
        os.remove(smi_path)
    else:
        xyz = param

    xyz_data = [l + '\n' for l in xyz.split('\n')[2:-1]]
    symbols = [l.split()[0] for l in xyz_data]
    coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
    connectivity = get_connectivity_info(xyz_data)

    job.meta['status'] = 'Running pipeline'
    job.save_meta()

    results = pipeline(symbols, coordinates, connectivity, use_threshold=False)

    job.meta['status'] = 'Plotting'
    job.save_meta()

    output = {
        "smiles": param if '\n' not in param else '.xyz file',
        "xyz": xyz,
        "graph": plot_homo_molecule(results[0]),
        "node_level": results[3][1],
        "is_atom": results[0].is_atom.bool().numpy(),
        "is_lp": results[0].is_lp.bool().numpy(),
        "is_bond": results[0].is_bond.bool().numpy(),
        "interactions": plot_interaction_matrix(*results[2]),
        "interaction_table": results[2][0],
        "interaction_predictions": results[3][2],
        "a2b_results": results[3][0],
        "a2b_index": results[0].a2b_index.numpy()
    }

    output["node_level"][:, 4:8] = softmax(output["node_level"][:, 4:8])

    y_mean = np.array([1.6833026e-05, 9.7693789e-01, 2.6718247e00, 3.6606803e00, 0, 0, 0, 0, 1.9150e00, ] + [0] * 7)
    y_std = np.array([0.34014496, 0.9994328, 2.0642664, 3.03501, 100, 100, 100, 100, 8.9718e-02, ] + [1] * 7)

    output["node_level"] *= y_std
    output["node_level"] += y_mean

    output["atom_graph"] = plot_homo_molecule_targets(
        results[0], output['node_level'][output['is_atom']]
    )

    # print(output['node_level'])
    # output["node_level"][:, -6:-3] = np.clip(output["node_level"][:, -6:-3], 0, 1)
    # output["node_level"][:, -6] = 1 - output["node_level"][:, -5] - output["node_level"][:, -4] - output["node_level"][:, -3] # Normalize to one

    output['interaction_predictions'] = transform_edge_predictions(
        output['interaction_predictions'],
        output['node_level'].shape[0],
        output['is_atom'].sum()
    )

    job.meta['status'] = 'Completed'
    job.save_meta()

    return output



if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
