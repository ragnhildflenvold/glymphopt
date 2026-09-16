import click
import json
import numpy as np


def json_serialize_entry(d):
    return {"point": list(d["point"]), "value": float(d["value"])}


@click.command()
@click.option("--alpha", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
def main(output, alpha, r):
    x = np.array([alpha, r])
    x_opt = np.array([1.5, 4e-7])
    error = {"point": x, "value": np.linalg.norm((x - x_opt) / x_opt) ** 2}
    with open(output, "w") as f:
        json.dump(json_serialize_entry(error), f, indent=2)


if __name__ == "__main__":
    main()
