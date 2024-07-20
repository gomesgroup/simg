import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert JSON to JSONL")
    parser.add_argument("--input", type=str, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, help="Path to the output JSONL file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully converted {args.input} to {args.output}")