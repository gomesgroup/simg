import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--n_parts', type=int)

    args = parser.parse_args()

    data = sum([
        torch.load(os.path.join(args.dir, file)) for file in tqdm(os.listdir(args.dir)) if
        file.endswith('_merged.pt')
    ], [])

    assignments = np.random.randint(0, args.n_parts, len(data))

    for part in range(args.n_parts):
        part_data = [g for g, idx in zip(data, assignments) if (idx == part) or (g.type != 'train')]

        torch.save(part_data, f'part_ensemble_{part}.pt')
