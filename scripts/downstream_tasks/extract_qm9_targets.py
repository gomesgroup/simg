import argparse
import gzip
import os
import json
import pickle as pkl

from tqdm.autonotebook import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm9_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    targets_data = {}

    for file in ['train', 'valid', 'test']:
        with gzip.open(os.path.join(args.qm9_path, file + '.jsonl.gz'), 'rt') as f:
            for line in tqdm(f):
                data = json.loads(line)
                targets_data[int(data['id'].split(':')[1])] = {
                    'targets': data['targets'],
                    'source': file
                }

    with open(args.output_path, 'wb') as f:
        pkl.dump(targets_data, f)
