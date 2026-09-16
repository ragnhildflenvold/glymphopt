import json
from typing import Any

import click
import numpy as np

from glymphopt.grid_scheduler import schedule_next_grid
from glymphopt.param_utils import encode_param_dict


def json_deserialize_history(d) -> dict[str, Any]:
    return {"point": np.array(d["point"]), "value": np.float64(d["error"])}


@click.command()
@click.option("--output", "-o", required=True)
@click.option("--param", nargs=3, multiple=True)
@click.option("--history")
def main(output, param, history):
    params = [p[0] for p in param]
    lower_bound = [float(p[1]) for p in param]
    upper_bound = [float(p[2]) for p in param]
    if history:
        with open(history, "r") as f:
            full_hist = [json_deserialize_history(d) for d in json.load(f)]
    else:
        full_hist = []

    combinations = []
    for param_vals in schedule_next_grid(lower_bound, upper_bound, 5, full_hist):
        combinations.append({key: val for key, val in zip(params, param_vals)})
    lines = [encode_param_dict(d, 5) for d in combinations]
    with open(output, "w") as f:
        for line in lines:
            f.write(line + "_error.json\n")


if __name__ == "__main__":
    main()
