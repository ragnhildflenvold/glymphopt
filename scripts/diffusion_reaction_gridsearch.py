import click
import pandas as pd

from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.minimize import adaptive_grid_search
from glymphopt.parameters import singlecomp_parameters, default_twocomp_parameters
from glymphopt.singlecompartment import SingleCompartmentInverseProblem
from glymphopt.utils import parse_evaluation


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--iter", "iterations", type=int, default=5)
@click.option("-dt", type=float, default=3600)
def main(input, output, iterations, dt):
    domain = read_mesh(input)
    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")
    coefficients = singlecomp_parameters(default_twocomp_parameters())

    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)

    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    problem = SingleCompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True
    )
    best, hist, full_hist = adaptive_grid_search(
        func=problem.F,
        lower_bounds=[0.1, 0.0],
        upper_bounds=[10.0, 1e-5],
        n_points_per_dim=5,
        n_iterations=iterations,
    )
    records = [parse_evaluation(pointeval, coeffconverter) for pointeval in full_hist]
    data = pd.DataFrame.from_records(records)
    data.to_csv(output, sep=";", index=False)


if __name__ == "__main__":
    main()
