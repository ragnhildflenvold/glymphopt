import numbers
from typing import Any

import pint

ureg = pint.get_application_registry()
dimless = pint.Unit("")
percent = 0.01 * dimless
mm = ureg.mm
cm = ureg.cm
s = ureg.second
minute = ureg.minute
liters = ureg.liters


DEFAULT_PARAMETERS = {
    "n": {  # Volume fractions
        "e": 20 * percent,
        "p": 2 * percent,
    },
    "D": {  # Diffusion coefficients
        "e": 1.3e-4 * mm**2 / s,
    },
    "t": {  # Transfer coefficient (permeability * area-volume ratio)
        "ep": 2.9e-2 * 1 / s,
        "pb": 2.0e-6 * 1 / s,
    },
    "k": {  # Surface membrane conductivity
        "e": 1.0e-5 * mm / s,
        "p": 3.7e-4 * mm / s,
    },
    "gamma": (20.0 * dimless),  # Factor of enhanced diffusion in PVS
    "rho": 0.113 * dimless,  # Ratio of free diffusion of gadobutrol to water
    "eta": 0.39 * dimless,
}


def get_default_parameters():
    default_parameters = {**DEFAULT_PARAMETERS}
    gamma = default_parameters["gamma"]
    De = default_parameters["D"]["e"]
    default_parameters["D"]["p"] = gamma * De
    return {**DEFAULT_PARAMETERS}


def get_dimless_parameters(T: pint.Quantity, X: pint.Quantity):
    default = get_default_parameters()
    n = default["n"]
    D = default["D"]
    t = default["t"]
    k = default["k"]
    gamma = default["gamma"]
    rho = default["rho"]
    eta = default["eta"]
    return {
        "n": {i: (ni).to("") for i, ni in n.items()},
        "D": {i: (T / X**2 * Di).to("") for i, Di in D.items()},
        "t": {ij: (T * t).to("") for ij, t in t.items()},
        "k": {i: (T / X * ki).to("") for i, ki in k.items()},
        "gamma": gamma,
        "rho": rho,
        "eta": eta,
    }


def default_twocomp_parameters(T: pint.Quantity = 1 * s, X: pint.Quantity = 1 * mm):
    default_parameters = remove_units(get_dimless_parameters(T, X))
    return flatten_dict(default_parameters)


def singlecomp_parameters(twocomp_parameters=None):
    twocomp_parameters = twocomp_parameters or default_twocomp_parameters()
    params = flatten_dict(twocomp_parameters)
    n_e = params["n_e"]
    n_p = params["n_p"]
    k_e = params["k_e"]
    k_p = params["k_p"]
    t_pb = params["t_pb"]
    gamma = params["gamma"]
    rho = params["rho"]
    eta = params["eta"]
    phi = n_e + n_p
    return {
        "a": (n_e + gamma * n_p) / phi,
        "r": t_pb / phi,
        "k": (k_e + k_p) / phi,
        "phi": phi,
        "eta": eta,
        "rho": rho,
    }


def flatten_dict(nested_dict, prefix=""):
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            flat_dict = flat_dict | flatten_dict(val, f"{key}_")
        else:
            flat_dict[f"{prefix}{key}"] = val
    return flat_dict


def remove_units(params):
    """Converts all quantities to a dimless number."""
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = remove_units(val)
        else:
            dimless[key] = val.magnitude
    return dimless


def print_quantities(p, offset, depth=0):
    """Pretty printing of dictionaries filled with pint.Quantities (or numbers)"""
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth * '  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if is_quantity(value):
                print(f"{depth * '  '}{str(key):<{format_size + 1}}: {value:.3e}")
            else:
                print(f"{depth * '  '}{str(key):<{format_size + 1}}: {value}")


def is_quantity(x: Any) -> bool:
    return isinstance(x, pint.Quantity) or isinstance(x, numbers.Complex)


def unpack(p, *args):
    return tuple(p[i] for i in args)


def consume(p, *args):
    return tuple(p.pop(i) for i in args)


if __name__ == "__main__":
    default = get_default_parameters()
    print_quantities(default, offset=4)
    print_quantities(singlecomp_parameters(default), offset=4)
