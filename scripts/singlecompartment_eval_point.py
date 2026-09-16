import json
import click
import numpy as np

from i2m.evaluation_data import scipy_sparse_csr_to_dolfin
from i2m.voxel_center_minimization import reconstruct_evaluation_matrix
from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_augmented_dti, read_function_data, read_mesh
from glymphopt.operators import matrix_operator
from glymphopt.parameters import (
    default_twocomp_parameters,
    singlecomp_parameters,
)
from glymphopt.singlecompartment import SingleCompartmentInverseProblem


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--eval", "-e", type=str, required=True)
@click.option("--dt", type=float, default=3600)
@click.option("--a", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--k", type=float)
@click.option("--phi", type=float)
def main(input, output, eval, dt, **kwargs):
    # Load default parameters, dimensionalize and overwrite coefficients
    default_coefficients = singlecomp_parameters(default_twocomp_parameters())
    overwrite_coefficients = {
        key: val for key, val in kwargs.items() if val is not None
    }
    coefficients = default_coefficients | overwrite_coefficients

    domain = read_mesh(input)
    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")

    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)
    problem = SingleCompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True, solver_type="iterative"
    )
    Y = problem.forward(coeffconverter.to_vector())
    Ym = problem.measure(Y)

    npzfile = np.load(eval)
    M = reconstruct_evaluation_matrix(npzfile)
    M_ = matrix_operator(scipy_sparse_csr_to_dolfin(M))
    C = [npzfile[label] for idx in range(5) if (label := f"vector{idx}") in npzfile]
    assert len(C) > 0, f"No vector data found in {eval}"
    error = 0.0
    for ym, c in zip(Ym[1:], C[1:]):
        e = M_(ym.vector())[:] - c
        error += e.dot(e) / c.dot(c)

    with open(output, "w") as f:
        json.dump(
            {"point": [coefficients["a"], coefficients["r"]], "error": error},
            f,
        )


if __name__ == "__main__":
    main()
