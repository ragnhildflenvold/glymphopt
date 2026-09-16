import json
from typing import Any

import click


@click.command()
@click.argument("inputs", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
@click.option("--history", type=click.Path(exists=True))
def main(inputs, output, history):
    if history:
        with open(history, "r") as f:
            try:
                records = json.load(f)
            except Exception as e:
                print("_" * 80)
                print("Input: ", history)
                print(f.read().split())
                print("_" * 80)
                raise e
    else:
        records = []

    for input in inputs:
        with open(input, "r") as f:
            try:
                record = json.load(f)
            except Exception as e:
                print("_" * 80)
                print("Input: ", input)
                print(f.read().split())
                print("_" * 80)
                raise e
            records.append(record)

    with open(output, "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
