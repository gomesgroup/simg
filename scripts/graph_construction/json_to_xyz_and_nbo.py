import json
import gzip
import argparse

from tqdm import tqdm


def extract_data(line: str):
    data = json.loads(line)

    molecular_data = data["output"]["initial_molecule"]["sites"]

    nbo_data = data["output"]["nbo"]
    nbo_str = json.dumps(nbo_data)

    xyz_lines = [str(len(molecular_data)), nbo_str]

    for atom in molecular_data:
        xyz_lines.append("\t".join([atom["name"]] + list(map(str, atom["xyz"]))))

    xyz_str = "\n".join(xyz_lines)
    id = data['task_label']

    return xyz_str, id


def process_file(f):
    for i, line in enumerate(tqdm(f)):
        xyz_str, id = extract_data(line)
        print(xyz_str, '\n', id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract xyz and nbo data from json lines file format")
    parser.add_argument("--path", type=str, help="Path to the json lines file")

    args = parser.parse_args()

    if args.path.endswith(".gz"):
        with gzip.open(args.path, "rt") as f:
            process_file(f)
    else:
        with open(args.path, "r") as f:
            process_file(f)
