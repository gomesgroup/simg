import uuid
import os
import pickle as pkl

from joblib import Parallel, delayed
from tqdm import tqdm


def convert2xyz(smi):
    path = uuid.uuid4().hex
    xyz_path = f"{path}.xyz"
    smi_path = f"{path}.smi"

    with open(smi_path, "w") as f:
        f.write(smi + '\n')

    os.system(f'obabel -i smi {smi_path} -o xyz -O {xyz_path} --gen3d >/dev/null 2>&1')

    with open(xyz_path, "r") as f:
        xyz = f.read()

    os.remove(xyz_path)
    os.remove(smi_path)

    return (smi, xyz)


all_smiles_splitted = [l.strip() for l in open('smiles.txt', 'r').readlines()]
all_smiles_splitted = [l for l in all_smiles_splitted if l]

xyzs = Parallel(n_jobs=128)(delayed(convert2xyz)(smi) for smi in tqdm(all_smiles_splitted))

with open('xyzs.pkl', 'wb') as f:
    pkl.dump(xyzs, f)
