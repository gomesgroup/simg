import pickle as pkl
import numpy as np
from tqdm import tqdm

from simg.model_utils import pipeline
from simg.data import get_connectivity_info

with open('xyzs.pkl', 'rb') as f:
    data = pkl.load(f)

smi, mol = data[0]

xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
symbols = [l.split()[0] for l in xyz_data]
coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
connectivity = get_connectivity_info(xyz_data)

for i in range(0, len(data), 10_000):
    mols = data[i: i + 10_000]
    output = []

    for smi, mol in tqdm(mols):
        try:
            xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
            symbols = [l.split()[0] for l in xyz_data]
            coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
            connectivity = get_connectivity_info(xyz_data)

            output.append([
                smi, mol, pipeline(symbols, coordinates, connectivity)
            ])
        except:
            continue

    with open(f'outputs/{i}.pkl', 'wb') as f:
        pkl.dump(output, f)
