import msgpack
from tqdm import tqdm


def sample_conformer(conformers):
    xyz = conformers[0]['xyz']
    els = [l[0] for l in xyz]
    return set(els)


def sample_molecules(mols):
    elements = set()
    for molecule in mols:
        conformers = mols[molecule]['conformers']
        els = sample_conformer(conformers)
        elements.update(els)

    return elements


unpacker = msgpack.Unpacker(open('/Users/daniilboiko/Downloads/geom_exp/drugs_crude.msgpack', 'rb'))

all_elements = set()
for i, molecules in tqdm(enumerate(unpacker), total=292):
    elements = sample_molecules(molecules)
    all_elements.update(elements)
    print(all_elements)
