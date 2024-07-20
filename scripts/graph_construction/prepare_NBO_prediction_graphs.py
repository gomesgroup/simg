import argparse
import yaml
import gzip
import logging

import torch
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed

from simg.graph_construction import construct_NBO_graph

logging.basicConfig(level=logging.INFO)

MODE_MAPPER = {
    "lps_bonds": construct_NBO_graph,
}


def get_data(path):
    data = []

    with gzip.open(path, "rt") as f:
        with tqdm() as t:
            is_number_of_atoms = True
            number_of_atoms = 0

            cache = []

            for line in f:
                if is_number_of_atoms:
                    number_of_atoms = int(line) + 2
                    is_number_of_atoms = False

                else:
                    cache.append(line)
                    number_of_atoms -= 1

                    if number_of_atoms == 0:
                        t.update()
                        data.append(cache)
                        cache = []

                        is_number_of_atoms = True

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare graphs for NBO prediction")

    parser.add_argument("--path", type=str, help="Path to the NBO data file")
    parser.add_argument("--configs", type=str, help="Path to the configs file")
    parser.add_argument("--output_path", type=str, help="Path to the output files")
    parser.add_argument("--n_jobs", type=int, default=16, help="Number of jobs to run in parallel")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Number of compounds to save in each batch")
    parser.add_argument("--mode", type=str, default="train", help="Graph generation mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        graph_construction_config = yaml.safe_load(f)

    data = get_data(args.path)

    if not args.debug:
        graphs = Parallel(n_jobs=args.n_jobs)(
            delayed(MODE_MAPPER[args.mode])(cmpd, graph_construction_config) for cmpd in tqdm(data))
    else:
        graphs = [construct_NBO_graph(cmpd, graph_construction_config) for cmpd in tqdm(data)]

    graphs = [g for g in graphs if g is not None]

    for i in range(0, len(graphs), args.batch_size):
        torch.save(graphs[i:i + args.batch_size], args.output_path + str(i // args.batch_size) + ".pt")
        logging.info(f"Saved {i} to {i + args.batch_size}")

    logging.info(f"Saved {len(graphs)} graphs")
