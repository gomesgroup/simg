import os
import msgpack

from tqdm import tqdm
from joblib import Parallel, delayed
import torch

from simg.data import get_connectivity_info
from simg.model_utils import get_initial_graph, predict_lps, get_final_graph, prepare_graph
from process_geom import process_xyz

unpacker = msgpack.Unpacker(open('drugs_crude.msgpack', 'rb'))
output_path = 'graphs/'


def process_conformer(conformer):
    xyz = conformer['xyz']
    xyz = process_xyz(xyz)

    symbols, coordinates = zip(*xyz)

    xyz_str = ['\t'.join([atom] + [str(x) for x in coords]) + '\n' for atom, coords in xyz]
    connectivity = get_connectivity_info(xyz_str)

    molecular_graph = get_initial_graph(symbols, coordinates, connectivity)
    n_lps, n_conj_lps = predict_lps(molecular_graph)
    graph = get_final_graph(molecular_graph, connectivity, n_lps, n_conj_lps)
    graph = prepare_graph(graph)

    return graph


for i, molecules in tqdm(enumerate(unpacker), total=292):
    output_dir = os.path.join(output_path, str(i))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (smiles, molecule) in enumerate(tqdm(molecules.items(), leave=False)):
        if molecule['charge'] != 0:
            continue

        conformers = molecule['conformers']

        graphs = Parallel(n_jobs=-1)(delayed(process_conformer)(conformer) for conformer in conformers)
        torch.save(graphs, os.path.join(output_dir, f'{i}.pt'))
