import msgpack
import random
import os
import numpy as np
from tqdm import tqdm

atomic_number2symbol = {
    1: 'H',
    5: 'B',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    13: 'Al',
    14: 'Si',
    15: 'P',
    16: 'S',
    17: 'Cl',
    33: 'As',
    35: 'Br',
    53: 'I',
    80: 'Hg',
    83: 'Bi',
}


def process_xyz(xyz):
    xyz = [(atomic_number2symbol[int(line[0])], line[1:]) for line in xyz]
    return xyz


def sample_conformer(conformers):
    weights = [c['boltzmannweight'] for c in conformers]

    weights = np.array(weights)
    weights /= weights.sum()

    idx = np.random.choice(len(conformers), p=weights)
    xyz = conformers[idx]['xyz']
    xyz = process_xyz(xyz)
    return idx, xyz


def sample_molecules(molecules, n):
    keys = list(molecules.keys())
    keys = sorted(keys)

    selected_keys = random.sample(list(range(len(keys))), n)

    output = []
    for key_idx in selected_keys:
        smiles = keys[key_idx]
        molecule = molecules[smiles]

        if molecule['charge'] != 0:
            continue

        conformers = molecule['conformers']
        sample_idx, sample_xyz = sample_conformer(conformers)

        output.append((f'{key_idx}_{sample_idx}', (smiles, sample_idx), sample_xyz))

    return output


def save_samples(samples, path, prefix):
    for sample_name, (smiles, _), sample_xyz in samples:
        with open(os.path.join(path, prefix + sample_name + '.xyz'), 'w') as f:
            f.write(str(len(sample_xyz)) + '\n')
            f.write(prefix + sample_name + '\t' + smiles + '\n')
            f.write('\n'.join(['\t'.join([atom] + [str(x) for x in coords]) for atom, coords in sample_xyz]) + '\n')


unpacker = msgpack.Unpacker(open('/Users/daniilboiko/Downloads/geom_exp/drugs_crude.msgpack', 'rb'))

for i, molecules in tqdm(enumerate(unpacker), total=292):
    samples = sample_molecules(molecules, 100)
    save_samples(samples, 'samples_100', str(i) + '_')
