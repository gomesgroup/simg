import logging
import os
from typing import List, Tuple
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from simg.utils import atom_type_one_hot
from simg.models import GNN_LP, GNN
from simg.data import block_diagonal, get_atom_atom_edges, get_connectivity_info

# get path for this file
THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

LP_CHECKPOINT_PATH = (
    f"{THIS_FILE_PATH}/checkpoints/lp_pred_model.ckpt"
)

INTERACTION_CHECKPOINT_PATH = (
    f"{THIS_FILE_PATH}/checkpoints/nbo_pred_model.ckpt"
)

lp_model = GNN_LP.load_from_checkpoint(LP_CHECKPOINT_PATH)
gnn = GNN.load_from_checkpoint(INTERACTION_CHECKPOINT_PATH)

lp_model.eval()
gnn.eval()

logging.basicConfig(level=logging.INFO)


def parse_xyz(xyz_path=None, sdf_file=None):
    if sdf_file is not None:
        with open(sdf_file, "r") as f:
            sdf_data = f.read()

        # split \n
        sdf_data = sdf_data.split("\n")
        n_atoms = int(float(sdf_data[3].split()[0]))

        symbols = [sdf_data[i].split()[3] for i in range(4, 4 + n_atoms)]
        coordinates = np.array([
            [float(sdf_data[i].split()[0]), float(sdf_data[i].split()[1]), float(sdf_data[i].split()[2])]
            for i in range(4, 4 + n_atoms)
        ])

        connectivity = get_connectivity_info(sdf_file=sdf_file)
    else:
        with open(xyz_path) as f:
            xyz_data = f.readlines()[2:]

        coordinates = np.array([[float(x) for x in line.split()[1:]] for line in xyz_data])
        symbols = [line.split()[0] for line in xyz_data]
        connectivity = get_connectivity_info(xyz_data=xyz_data)

    return symbols, coordinates, connectivity


def parse_sdf_v3000(file_path):
    with open(file_path, "r") as f:
        sdf_data = f.read()

    # split \n
    sdf_data = sdf_data.split("\n")

    # get number of atoms
    n_atoms = int(float(sdf_data[5].split()[3]))
    n_bonds = int(float(sdf_data[5].split()[4]))

    # get coordinates
    coordinates = np.array([
        [float(sdf_data[i].split()[4]), float(sdf_data[i].split()[5]), float(sdf_data[i].split()[6])]
        for i in range(7, 7 + n_atoms)
    ])

    # get symbols
    symbols = [sdf_data[i].split()[3] for i in range(7, 7 + n_atoms)]

    # get connectivity

    connectivity = []
    for i in range(7 + n_atoms + 2, 7 + n_atoms + n_bonds + 2):
        atom_A = int(sdf_data[i].split()[4]) - 1
        atom_B = int(sdf_data[i].split()[5]) - 1
        bond_type = int(sdf_data[i].split()[3])
        connectivity.append((atom_A, atom_B, bond_type))

    return symbols, coordinates, connectivity


def get_bond_features(coordinates, connectivity):
    bond_lengths = [
        np.linalg.norm(coordinates[atom_A] - coordinates[atom_B])
        for atom_A, atom_B, _ in connectivity
        for _ in range(2)
    ]

    bond_types = np.zeros((len(connectivity), 4))
    for i, (_, _, order) in enumerate(connectivity):
        bond_types[i, order - 1] = 1

    bond_types = np.vstack([bond_types, bond_types])

    all_bond_features = np.hstack(
        [bond_types, np.array(bond_lengths).reshape(len(connectivity) * 2, 1)]
    ).astype(float)
    return all_bond_features


def get_atom_participates_bond(connectivity, last_index):
    atom_bond_indexes = []
    cum_j = 0
    for i, (atom_A, atom_B, order) in enumerate(connectivity):
        for j in range(order):
            if j:
                cum_j += 1
            bond_node = last_index + i + cum_j
            atom_bond_indexes.append([atom_A, bond_node])
            atom_bond_indexes.append([atom_B, bond_node])

    # duplicate list with inverted elements
    atom_bond_indexes = atom_bond_indexes + [[j, i] for i, j in atom_bond_indexes]

    return atom_bond_indexes


def get_distance(xyz_1, xyz_2):
    return np.linalg.norm(xyz_1 - xyz_2)


def get_initial_graph(symbols, coordinates, connectivity):
    # one hot encoding of atoms
    x = torch.tensor([atom_type_one_hot(atom) for atom in symbols], dtype=torch.float32)

    # bond features
    edge_attr = torch.tensor(
        get_bond_features(coordinates, connectivity), dtype=torch.float32
    )

    # remove last column (distances do not matter for LP inference)
    edge_attr = edge_attr[:, :-1]

    # turn into bidirectional graph
    connectivity = connectivity + [(j, i, k) for i, j, k in connectivity]

    # Create graph as coordination list
    edge_index = torch.tensor([[x[0], x[1]] for x in connectivity]).t()

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        symbols=symbols,  # this is not used for LP inference
        xyz_data=torch.tensor(
            np.array(coordinates)
        ),  # this is not used for LP inference
        num_nodes=len(symbols),
    )

    return graph


def predict_lps(mol_graph):
    with torch.no_grad():
        pred = lp_model(mol_graph)

    n_lps = pred[:, :5].argmax(dim=1).tolist()
    n_conj_lps = pred[:, 5:].argmax(dim=1).tolist()

    for i, (n_lp, conj_lp, atom) in enumerate(
            zip(n_lps, n_conj_lps, mol_graph.symbols)
    ):
        if atom == "H" and n_lp > 0:
            n_lps[i] = 0
            n_conj_lps[i] = 0
        if conj_lp > n_lp:
            n_conj_lps[i] = n_lp

    return n_lps, n_conj_lps


def get_atoms_per_lp(n_lps_per_atom):
    atoms_per_lp = []
    for i, n_lp in enumerate(n_lps_per_atom):
        atoms_per_lp.extend([i] * n_lp)
    return atoms_per_lp


def get_final_graph(molecular_graph, connectivity, n_lps_per_atom, n_conj_lps):
    N_ATOMS = molecular_graph.num_nodes
    N_LPS = sum(n_lps_per_atom)
    N_BONDS = sum([x[-1] for x in connectivity])

    N_ATOM_ATOM_FEAT = 4
    N_ATOM_LP_FEAT = 7
    N_ATOM_BOND_FEAT = 2
    N_EDGE_TYPE_FEAT = 3

    N_ATOM_TYPE = 16
    N_LP_TYPE = 2
    N_ORBITAL_FEAT = 2

    # this is equivalent to lone_pair_data["atoms_list"] in data.py
    atoms_per_lp = get_atoms_per_lp(n_lps_per_atom)

    features = {}
    features["atom"] = molecular_graph.x
    features["bond"] = molecular_graph.edge_attr
    features["lp"] = features["atom"][atoms_per_lp]

    # get a bool list of lone pairs that p-s<80 this assumes that
    # the firsts lone pairs of and atom are the right ones. but
    # that doesn't matter since there is no way to determine which
    # ones are based on the quantity.
    lp_is_conj = []
    aux = get_atoms_per_lp(n_conj_lps)

    for i in atoms_per_lp:
        if i in aux:
            lp_is_conj.append(1)
            aux.remove(i)
        else:
            lp_is_conj.append(0)

    # This is to reorder the conjugated lone pairs to be the last ones, just to match the original code (in most cases)

    # get all values in atoms_per_lp, without duplicates
    atoms = list(set(atoms_per_lp))
    new_lp_is_conj = []
    for atom in atoms:
        # get the indexes in lp_is_conj that correspond to the current atom
        indexes = [i for i, x in enumerate(atoms_per_lp) if x == atom]

        vals = [lp_is_conj[i] for i in indexes]
        # invert list
        new_lp_is_conj.extend(vals[::-1])

    lp_is_conj = new_lp_is_conj

    if N_LPS != 0:
        features["lp"] = np.hstack(
            (
                features["lp"],
                np.array(lp_is_conj)[:, None],
                np.array([atoms_per_lp.count(atom) for atom in atoms_per_lp])[:, None],
            )
        )

    separated_connectivity = []
    order_one_hot = []
    for atom_A, atom_B, order in connectivity:
        for _ in range(order):
            separated_connectivity.append([atom_A, atom_B])
            # get one hot for order, there are 4 types
            one_hot = [0, 0, 0, 0]
            one_hot[order - 1] = 1
        order_one_hot.append(one_hot)
        order_one_hot.append(one_hot)  # duplicate for bidirectional graph

    xyz_data = molecular_graph.xyz_data.tolist()
    xyz_data += [
        xyz for i, xyz in enumerate(xyz_data) for _ in range(n_lps_per_atom[i])
    ]
    xyz_data += [
        ((np.array(xyz_data[a]) + np.array(xyz_data[b])) / 2).tolist()
        for a, b in separated_connectivity
    ]
    xyz_data = torch.FloatTensor(xyz_data)

    bond_lengths = [
        get_distance(np.array(xyz_data[a]), np.array(xyz_data[b])).tolist()
        for a, b in separated_connectivity
    ]
    is_pi_bond = []
    last = (None, None, None)
    for i in separated_connectivity:
        if i == last:
            is_pi_bond.append(1)
        else:
            is_pi_bond.append(0)
        last = i

    features["orbital"] = np.vstack([bond_lengths, is_pi_bond]).T

    vector_data = [[0, 0, 0]] * (N_ATOMS + N_LPS)
    vector_data += [
        (np.array(xyz_data[a]) - np.array(xyz_data[b])).tolist()
        for a, b in separated_connectivity
    ]
    vector_data = torch.FloatTensor(vector_data)

    # ------------------------EDGES-------------------------

    edges = {}
    edges[("atom", "connects", "atom")] = get_atom_atom_edges(connectivity)

    edges[("atom", "has", "lone_pair")] = []
    edges[("lone_pair", "relates_to", "atom")] = []
    for lone_pair_idx, atom_idx in enumerate(atoms_per_lp):
        edges[("atom", "has", "lone_pair")].append(
            [atom_idx, lone_pair_idx + len(features["atom"])]
        )
        edges[("lone_pair", "relates_to", "atom")].append(
            [lone_pair_idx + len(features["atom"]), atom_idx]
        )

    # interaction between LP and BONDS
    interaction_edge_index = [
        (i, j)
        for i in range(N_ATOMS, N_ATOMS + N_LPS + N_BONDS)
        for j in range(N_ATOMS, N_ATOMS + N_LPS + N_BONDS)
        if i != j
    ]

    # --------------------GRAPH CONSTRUCTION-----------------

    is_atom = [1] * N_ATOMS + [0] * N_LPS + [0] * N_BONDS
    is_lp = [0] * N_ATOMS + [1] * N_LPS + [0] * N_BONDS
    is_bond = [0] * N_ATOMS + [0] * N_LPS + [1] * N_BONDS

    graph = Data()
    graph.x = block_diagonal(features["atom"], features["lp"], N_ATOM_TYPE + N_ATOM_TYPE + N_LP_TYPE)
    graph.x = block_diagonal(graph.x, features["orbital"], N_ATOM_TYPE + N_ATOM_TYPE + N_LP_TYPE + N_ORBITAL_FEAT)
    graph.x = torch.FloatTensor(graph.x)
    graph.x = torch.cat(
        (graph.x, torch.FloatTensor([is_atom, is_lp, is_bond]).T), dim=1
    )

    graph.xyz_data = xyz_data
    graph.vector_data = vector_data

    atom_participates_bond = get_atom_participates_bond(connectivity, N_ATOMS + N_LPS)
    graph.edge_index = sum([v for k, v in edges.items()], []) + atom_participates_bond
    graph.edge_index = torch.LongTensor(graph.edge_index).T

    graph.a2b_index = torch.LongTensor(
        atom_participates_bond[: len(atom_participates_bond) // 2]
    ).T
    graph.interaction_edge_index = torch.LongTensor(interaction_edge_index).T

    # -- EDGE_ATTR

    orbital_feat = []
    for feat in features["orbital"]:
        orbital_feat.append(feat)
        orbital_feat.append(feat)  # duplicate for bidirectional graph

    atom_atom = np.array(order_one_hot)
    atom_lp = np.vstack((features['lp'], features['lp']))  # duplicate for bidirectional graph
    atom_bond = np.vstack((orbital_feat, orbital_feat))  # duplicate for bidirectional graph

    b1 = block_diagonal(atom_atom, atom_lp, N_ATOM_ATOM_FEAT + N_ATOM_TYPE + N_LP_TYPE)
    b2 = block_diagonal(b1, atom_bond, N_ATOM_ATOM_FEAT + N_ATOM_TYPE + N_LP_TYPE + N_ATOM_BOND_FEAT)

    type_1 = np.array([1] * len(atom_atom) + [0] * len(atom_lp) + [0] * len(atom_bond))
    type_2 = np.array([0] * len(atom_atom) + [1] * len(atom_lp) + [0] * len(atom_bond))
    type_3 = np.array([0] * len(atom_atom) + [0] * len(atom_lp) + [1] * len(atom_bond))

    # put everything together
    graph.edge_attr = np.column_stack((b2, type_1, type_2, type_3))
    graph.edge_attr = torch.FloatTensor(graph.edge_attr)

    graph.symbol = molecular_graph.symbols + ["LP"] * N_LPS + ["BND"] * N_BONDS
    graph.is_atom = torch.FloatTensor(is_atom)
    graph.is_lp = torch.FloatTensor(is_lp)
    graph.is_bond = torch.FloatTensor(is_bond)

    return graph


def prepare_graph(graph):
    graph.edge_index = graph.edge_index.T
    graph.edge_index = [[int(num) for num in item] for item in list(graph.edge_index)]

    graph = Data(
        x=torch.FloatTensor(graph.x),
        edge_index=torch.LongTensor(graph.edge_index).T,
        edge_attr=torch.FloatTensor(graph.edge_attr),
        symbol=graph.symbol,
        is_atom=torch.Tensor(graph.is_atom),
        is_lp=torch.Tensor(graph.is_lp),
        is_bond=torch.Tensor(graph.is_bond),
        interaction_edge_index=torch.LongTensor(graph.interaction_edge_index),
        xyz_data=torch.FloatTensor(graph.xyz_data),
        vector_data=torch.FloatTensor(graph.vector_data),
        a2b_index=torch.LongTensor(graph.a2b_index).T,
    )

    return graph


def make_preds_no_gt(data, threshold=0, use_threshold=True):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    xyz_data, vector_data = data.xyz_data, data.vector_data
    a2b_index = data.a2b_index

    logging.info('Started indexing. Might take a while for big molecules')
    interaction_edge_index = [[i, j] for i in range(len(x)) for j in range(len(x))]
    interaction_edge_index = torch.LongTensor(interaction_edge_index).T

    logging.info('Started making predictions')
    with torch.no_grad():
        preds, a2b_preds, node_preds, int_preds, intermediate, intermediate_pred = gnn.forward(
            x, edge_index, edge_attr, data.interaction_edge_index, interaction_edge_index,
            xyz_data, vector_data, a2b_index
        )

    logging.info('Finished making predictions')

    preds = torch.sigmoid(preds.reshape((len(x), len(x)))).cpu().detach().numpy()

    n_atoms = int(data.is_atom.sum())
    preds = preds[n_atoms:, n_atoms:]
    np.fill_diagonal(preds, 0)

    if use_threshold:
        preds[preds < threshold] = 0
        preds[preds > 0] = 1

        all_idx = data.interaction_edge_index.T.numpy()

        row_idx, col_idx = np.where(preds == 1)
        orbital_interactions = np.concatenate((row_idx[:, None], col_idx[:, None]), axis=1) + n_atoms

        if np.shape(orbital_interactions)[0] > 1000:
            logging.info('Too many interactions, might take a while to process')
            # divide orbital interactions into chunks of size 500
            chunks = np.array_split(orbital_interactions, np.shape(orbital_interactions)[0] // 500)

            def get_int_idx(chunk):
                return np.where((all_idx[:, None] == chunk).all(axis=2))[0]

            int_idx = Parallel(n_jobs=4)(delayed(get_int_idx)(chunk) for chunk in tqdm(chunks))
            int_idx = np.concatenate(int_idx)

        else:
            int_idx = np.where((all_idx[:, None] == orbital_interactions).all(axis=2))[0]

        right_edges = torch.LongTensor(orbital_interactions.tolist()).T

        int_preds = int_preds[int_idx]
        data.interaction_edge_index = right_edges

    symbol = data.symbol[n_atoms:]
    index = np.arange(len(x))[n_atoms:]

    return (preds, symbol, index), (a2b_preds.cpu().detach().numpy(),
                                    node_preds.cpu().detach().numpy(),
                                    int_preds.cpu().detach().numpy())


def get_orbital_connectivity(graph) -> List[Tuple[int, int, int]]:
    """Get the connectivity of the graph based on bond orbitals.

    Different from the connectivity in the original graph, this function
    only considers the bond orbitals, not the bond order given by OpenBabel.
    This function can be employed when you want to compare the same molecule
    that have different bond orders (resonance structures).    
    """

    connectivity = []
    for edge, attr in zip(graph.edge_index.T, graph.edge_attr):
        is_atom_bond = bool(attr[-1])
        if is_atom_bond:
            connectivity.append((edge[0].item(), edge[1].item()))

    # order tuples where i<j
    for n, (i, j) in enumerate(connectivity):
        if i > j:
            connectivity[n] = (j, i)

    # remove duplicates
    connectivity = list(set(connectivity))

    # order by first element, then by second element in the same operation
    connectivity.sort(key=lambda x: (x[0], x[1]))

    bonds = []
    for n, (i, j) in enumerate(connectivity):
        for k, l in connectivity[n + 1:]:
            if j == k:
                bonds.append((i, l))
            if j == l:
                bonds.append((i, k))

    # create a list of tuples from bonds. (a,b,c) where c is the frequency of (a,b) in the list
    connectivity = [(i, j, bonds.count((i, j))) for i, j in bonds]

    # remove duplicates
    connectivity = list(set(connectivity))
    connectivity.sort(key=lambda x: (x[0], x[1]))

    return connectivity


def get_sdf_block(graph):
    connectivity = get_orbital_connectivity(graph)

    n_atoms = int(graph.is_atom.sum().item())
    atom_symbols = graph.symbol[:n_atoms]
    xyz_data = graph.xyz_data[:n_atoms]

    # do the same as above but store everything in just a string
    sdf_lines = ""
    sdf_lines += "TEMPORARY\n\n\n"
    # write counter line
    sdf_lines += "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n".format(n_atoms, len(connectivity))

    for xyz, symbol in zip(xyz_data, atom_symbols):
        sdf_lines += "{:>10.4f}{:>10.4f}{:>10.4f} {}   0  0  0  0  0  0  0  0  0  0  0  0\n".format(xyz[0], xyz[1],
                                                                                                    xyz[2], symbol)

    for i, (a, b, order) in enumerate(connectivity):
        sdf_lines += "{:>3}{:>3}{:>3}  0  0  0  0\n".format(a + 1, b + 1, order)

    sdf_lines += "M  END\n$$$$"

    return sdf_lines


def pipeline(symbols, coordinates, connectivity, threshold=0, use_threshold=True):
    molecular_graph = get_initial_graph(symbols, coordinates, connectivity)
    n_lps, n_conj_lps = predict_lps(molecular_graph)
    graph = get_final_graph(molecular_graph, connectivity, n_lps, n_conj_lps)
    graph = prepare_graph(graph)
    (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds) = make_preds_no_gt(graph, threshold, use_threshold)

    return graph, (n_lps, n_conj_lps), (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds)
