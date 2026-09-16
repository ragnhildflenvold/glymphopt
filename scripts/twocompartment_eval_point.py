import click
import json
import dolfin as df
import numpy as np

from i2m.evaluation_data import scipy_sparse_csr_to_dolfin
from i2m.voxel_center_minimization import reconstruct_evaluation_matrix
from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_augmented_dti, read_function_data, read_mesh
from glymphopt.measure import LossFunction, MRILoss
from glymphopt.operators import bilinear_operator, mass_matrix, matrix_operator
from glymphopt.parameters import default_twocomp_parameters
from glymphopt.twocompartment import MulticompartmentInverseProblem
from glymphopt.utils import with_suffix


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--eval", "-e", type=str, required=True)
@click.option("--dt", type=float, default=3600)
@click.option("--gamma", type=float)
@click.option("--t_ep", "t_ep", type=float)
@click.option("--t_pb", "t_pb", type=float)
@click.option("--k_e", "k_e", type=float)
@click.option("--k_p", "k_p", type=float)
@click.option("--visual", is_flag=True)
def main(input, output, eval, dt, visual, **kwargs):
    # Load default parameters, dimensionalize and overwrite coefficients
    default_coefficients = default_twocomp_parameters()
    overwrite_coefficients = {
        key: val for key, val in kwargs.items() if val is not None
    }
    coefficients = default_coefficients | overwrite_coefficients

    domain = read_mesh(input)
    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(input, domain, "concentration")

    td, Y_bdry_tmp = read_function_data(input, domain, "boundary_concentration")
    # Want Y_bdry in same function space as Yd (don't remember why its needed)
    Y_bdry = [
        df.Function(Yd[0].function_space(), name="boundary_concentration")
        for _ in Y_bdry_tmp
    ]
    [Y_bdry[i].interpolate(Y_bdry_tmp[i]) for i in range(len(td))]

    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)

    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))
    problem = MulticompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True
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
            {"point": [coefficients["gamma"], coefficients["t_pb"]], "error": error},
            f,
        )


if __name__ == "__main__":
    main()
