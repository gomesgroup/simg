import argparse
import torch
import random

from tqdm.autonotebook import tqdm


def get_state(i, train_slpit, val_split):
    if i < len(data) * train_slpit:
        return "train"
    elif i < len(data) * (1 - val_split):
        return "val"
    else:
        return "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str)

    args = parser.parse_args()

    data = torch.load(args.graphs_path)

    random.seed(42)
    random.shuffle(data)

    for i, graph in tqdm(enumerate(data)):
        graph.type = get_state(i, 0.5, 0.25)

    torch.save(
        data, args.graphs_path + "_merged.pt"
    )
