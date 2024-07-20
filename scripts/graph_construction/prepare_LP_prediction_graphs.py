import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from joblib import Parallel, delayed

NUM_ATOM_FEATURES = 16
NUM_BOND_FEATURES = 4
NUM_LP_FEATURES = 19


def get_lp_graph(i, nbo):
    num_atoms = nbo.is_atom.sum().int().item()

    nbo.edge_index = nbo.edge_index.T

    # get index of first LP or BND element
    num_lps = nbo.is_lp.sum()
    # get indexes where value of list is 1 (i.e. get the lp indexes)
    lp_indexes = np.where(nbo.is_lp == 1)[0]
    # get elements that contains any of lp_indexes
    lp_edge_index = np.where(np.isin(nbo.edge_index, lp_indexes))[0]
    # auxiliary list to get the lp_edge_index
    temp = [set(nbo.edge_index[j].tolist()) for j in lp_edge_index]
    # remove duplicates
    lp_edge_index = [list(item) for item in set(frozenset(item) for item in temp)]
    lp_edge_index = [sorted(item) for item in lp_edge_index]  # order each list

    # TASK 1:
    # get the number of lps per atom, in a list of length num_atoms
    num_lps = num_lps.int().item()
    n_lps_per_atom = torch.zeros(num_atoms, dtype=int)
    for i in lp_edge_index:
        n_lps_per_atom[i[0]] += 1

    # from node feature diagonal block, get only the feature indicating the number of conjugated LPs (index 5)
    conjugated = nbo.x[
                 num_atoms: num_atoms + num_lps,
                 NUM_ATOM_FEATURES: NUM_ATOM_FEATURES + NUM_LP_FEATURES,
                 ][:,
                 NUM_ATOM_FEATURES]  # THE INDEX IS NUM_ATOM_FEATURES BECAUSE THE LP FEATURES ARE APPENDED TO ONE-HOT ATOMS

    # TASK 2:
    # group conjugated LPs by lp_edge_index
    n_conjugated_per_atom = torch.zeros(num_atoms, dtype=int)
    ref = 0
    for j, num in enumerate(n_lps_per_atom):
        n_conjugated_per_atom[j] = conjugated[ref: ref + num].sum()
        ref += num

    # append n_lps_per_atom in each row of mol_graphs[i].x
    # concatenate _lp features with num of conjugated lps
    y = torch.stack((n_lps_per_atom, n_conjugated_per_atom)).T

    # get molecular features from nbo graphs
    x = nbo.x[:num_atoms, :NUM_ATOM_FEATURES]
    symbol = nbo.symbol[:num_atoms]
    mol_edge_index = []
    mol_edge_attr = []
    # remove all edges that correspond to LP or BND, and keep only atoms
    for j, coord in enumerate(nbo.edge_index):
        # check if coord contains a number from 0 to num_atoms-1
        if all(0 <= coord[a] < num_atoms for a in range(2)):
            mol_edge_index.append(coord)
            mol_edge_attr.append(nbo.edge_attr[j].tolist()[:NUM_BOND_FEATURES])

    # this is just like a molecular graph, but with y from the collected LP tasks
    lp_graph = Data(
        # x=torch.tensor(x).float(),
        x=x,
        y=y,
        edge_index=torch.stack(mol_edge_index).long().T,
        edge_attr=torch.tensor(mol_edge_attr).float(),
        num_nodes=num_atoms,
        symbol=symbol,
        qm9_id=nbo.qm9_id,
        type=nbo.type
    )

    return lp_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nbo_path", type=str, help="Path to the file with the NBO graph *.pt"
    )
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    nbo_graphs = torch.load(args.nbo_path)

    lp_graphs = Parallel(n_jobs=args.num_workers)(
        delayed(get_lp_graph)(i, nbo) for i, nbo in enumerate(tqdm(nbo_graphs)))

    torch.save(lp_graphs, args.output_path)
