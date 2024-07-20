import os
import argparse
import pickle as pkl

import torch
import pandas as pd

from data import split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    # Check input
    assert os.path.exists(args.input_file), "Input file does not exist"
    assert os.path.exists(args.splits_path), "Splits path does not exist"

    data = torch.load(args.input_file)
    train, val, test = split(data, args.splits_path)

    feature_dicts = pd.DataFrame([item.y for item in train])
    normalizers = [feature_dicts.mean().to_dict(), feature_dicts.std().to_dict()]
    with open(args.output_file, "wb") as f:
        pkl.dump(normalizers, f)
