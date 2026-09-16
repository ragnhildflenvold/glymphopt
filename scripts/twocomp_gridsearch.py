import click
import dolfin as df
import pandas as pd

from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.minimize import adaptive_grid_search
from glymphopt.parameters import default_twocomp_parameters
from glymphopt.twocompartment import MulticompartmentInverseProblem
from glymphopt.utils import parse_evaluation


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--iter", "iterations", type=int, default=10)
@click.option("--dt", type=float, default=3600)
def main(input, output, iterations, dt):
    coefficients = default_twocomp_parameters()
    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))

    domain = read_mesh(input)
    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry_tmp = read_function_data(input, domain, "boundary_concentration")

    # Want Y_bdry in same function space as Yd
    Y_bdry = [
        df.Function(Yd[0].function_space(), name="boundary_concentration")
        for _ in Y_bdry_tmp
    ]
    [Y_bdry[i].interpolate(Y_bdry_tmp[i]) for i in range(len(td))]

    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)
    problem = MulticompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True
    )

    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))
    best, hist, full_hist = adaptive_grid_search(
        func=problem.F,
        lower_bounds=[1.0, 0.0],
        upper_bounds=[45, 1e-5],
        n_points_per_dim=5,
        n_iterations=iterations,
    )
    records = [parse_evaluation(pointeval, coeffconverter) for pointeval in full_hist]
    data = pd.DataFrame.from_records(records)
    data.to_csv(output, sep=";", index=False)


if __name__ == "__main__":
    main()
