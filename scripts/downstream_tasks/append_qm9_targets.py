import argparse
import torch
import pickle as pkl

from tqdm.autonotebook import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_path", type=str)
    parser.add_argument("--graphs_path", type=str)

    args = parser.parse_args()

    with open(args.targets_path, 'rb') as f:
        targets_data = pkl.load(f)

    data = torch.load(args.graphs_path)

    for graph in tqdm(data):
        if graph.qm9_id not in targets_data:
            graph.type = 'remove'
            continue

        graph.normalized_targets = targets_data[graph.qm9_id]['targets']
        graph.type = targets_data[graph.qm9_id]['source']

    torch.save(
        data, args.graphs_path + "_merged.pt"
    )
