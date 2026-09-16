import itertools
import re
from functools import partial
from typing import Mapping, Optional, Sequence


def float_string_formatter(x: float, decimals: int):
    """Converts a numbers to a path-friendly floating points format without punctuation.
    ex: (1.3344, 3) -> 1334e-3"""
    try:
        if float(x) == float("inf"):
            return "inf"
    except TypeError:
        raise TypeError(f"Can't convert {x} to 'float'")

    return f"{x * 10 ** (-decimals):{f'.{decimals}e'}}".replace(".", "")


def encode_param_dict(d: Mapping[str, float], decimals: int) -> str:
    """Convert parameter-dictionary to string_format
    ex: {"a": 1.3344, "b": 1.0} -> a1334e-3_b1000e-3"""
    fsformat = partial(float_string_formatter, decimals=decimals)
    try:
        key_val_pairs = [f"{key}{fsformat(val)}" for key, val in sorted_dict(d).items()]
    except TypeError as e:
        print(f"Error while parsing dictionary:", d)
        raise e
    return "_".join(key_val_pairs)


def param_string_regex():
    param_re = r"[\w]+"
    float_re = r"-*\d+e[\+|-]\d+|inf"
    return re.compile(rf"_*(?P<param>{param_re}?)(?P<value>{float_re})_*")


def parse_param_string(param_string):
    """Converts parameter-string to dictionary of parameters
    ex: a1334e-3_b1000e-3 -> {"a": 1.3344, "b": 1.0}"""
    param_string_re = param_string_regex()
    coefficients = {}
    for match in param_string_re.finditer(param_string):
        param, value = match.groups()
        coefficients[param] = float(value)
    return coefficients


def to_scientific(num: float | str, decimals: int) -> str:
    """Convert floating-point number to latex-friendly floating-point format
    for e.g. plot-labelsh plotting: (0.0032, 2) -> 3.20\\times10^{-3}"""
    floatnum = float(num)
    if floatnum == float("inf"):
        return r"\infty"
    x = f"{floatnum:{f'.{decimals}e'}}"
    m = re.search(r"(\d\.{0,1}\d*)e([\+|\-]\d{2})", x)
    if m is None:
        raise ValueError(f"Regex expression not found in {x}")
    return f"{m.group(1)}\\times10^{{{int(m.group(2))}}}"


def create_param_variations(
    variations: Mapping[str, Sequence[float]],
    baseline: Mapping[str, float],
    model_params: Optional[Sequence[str]] = None,
) -> list[dict[str, float]]:
    """Create list of parameter sets, given baseline parameter dict, and a
    dictionary of all values to be tested for each of the parameters. If
    model_params are given, then any parameter not in the list are set to the
    baseline values.
    ex: ({"a": [1, 2], "c": [1, 2]}, {"a": 0, "b": 0, "c: 0}, ["a", "b"])
        -> [ {"a": 1, "b": 0, "c": 0}, {"a": 2, "b": 0, "c": 0} ]
    """
    if model_params is None:
        model_params = list(baseline.keys())

    products = itertools.product(
        *[variations[key] for key in variations]  # if key in model_params]
    )
    param_settings = []
    for product in products:
        new_setting = {**baseline}
        for key, val in zip(variations, product):
            if key in model_params:
                new_setting[key] = val
        param_settings.append(new_setting)
    return param_settings


def param_variation_strings(
    variation: Mapping[str, Sequence[float]],
    baseline: Mapping[str, float],
    decimals: int,
    model_params: Optional[Sequence[str]] = None,
) -> list[str]:
    """Takes a list of parameter-variations and the base-parameter setting,
    and creates a list of 2-decimal string-represented parameter sets.
    ex: ({"a": [1, 2]}, {"a": 0, "b": 0}, 2) -> ["a100e-2_b000e-2", "a200e-2_b000e-2"]
    """
    param_dicts = create_param_variations(variation, baseline, model_params)
    return list(
        set([encode_param_dict(param_dict, decimals) for param_dict in param_dicts])
    )


def param_set_reduction(
    paramdict: Mapping[str, float],
    model_params: Sequence["str"],
    baseline_parameters: Mapping[str, float],
) -> dict[str, float]:
    out = {**baseline_parameters}
    for key in model_params:
        out[key] = paramdict[key]
    return out


def param_dict_to_options(params, prefix="--"):
    return " ".join([f"{prefix}{key} {val}" for key, val in params.items()])


def sorted_dict(d):
    return {key: d[key] for key in sorted(d.keys())}


def flatten_dict(nested_dict, start=0):
    result = {}
    for key, value in nested_dict.items():
        if isinstance(value, list):
            for i, item in enumerate(value, start=start):
                result[f"{key}_{i}"] = item
        else:
            result[key] = value
    return result
