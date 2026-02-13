import os

from tqdm import tqdm
import numpy as np

from ase.io import read
from dscribe.descriptors import SOAP

QM7_GRAPH_PATH = "/home/dboiko/SPAHM/SPAHM/1_QM7/xyz"

atomic_symbols = ["H", "C", "N", "O", "F", "S"]


def xyz_to_soap(xyz_path):
    soap = SOAP(
        species=atomic_symbols,
        periodic=False,
        r_cut=5.0,
        n_max=4,
        l_max=4,
        average="inner",
    )

    ase_mol = read(xyz_path)
    soap_descriptors = soap.create(ase_mol)

    return soap_descriptors


files = sorted(os.listdir(QM7_GRAPH_PATH))
descriptors = []

print(files[:10])
for file in tqdm(files):
    if not file.endswith('.xyz'):
        continue

    descriptors.append(
        xyz_to_soap(os.path.join(QM7_GRAPH_PATH, file))
    )

descriptors = np.stack(descriptors)
print(descriptors.shape)

np.save(
    'soap_qm7.npy',
    descriptors
)
