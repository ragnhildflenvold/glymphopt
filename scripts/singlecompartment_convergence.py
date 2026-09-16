import time as pytime
import click
import dolfin as df
from loguru import logger
import numpy as np
import pandas as pd
import pantarei as pr

from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_augmented_dti, read_function_data, read_mesh
from glymphopt.measure import LossFunction, MRILoss
from glymphopt.operators import bilinear_operator, matrix_operator
from glymphopt.parameters import (
    default_twocomp_parameters,
    singlecomp_parameters,
)
from glymphopt.singlecompartment import SingleCompartmentInverseProblem
from glymphopt.utils import with_suffix


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--eval", "-e", type=str, required=True)
@click.option("--dt", type=float, default=3600)
@click.option("--a", type=float)
@click.option("--r", type=float)
@click.option("--k", type=float)
@click.option("--phi", type=float)
@click.option("--visual", is_flag=True)
def main(input, output, eval, dt, visual, **kwargs):
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
    try:
        Y = problem.forward(coeffconverter.to_vector())
    except RuntimeError as e:
        if problem.solver_type != "direct":
            raise e
        logger.warning("Direct solver failed, falling back to iterative.")
        problem.solver_type = "iterative"
        Y = problem.forward(coeffconverter.to_vector())

    Ym = problem.measure(Y)
    with df.HDF5File(domain.mpi_comm(), output, "w") as hdf:
        for ti, Ym_i in zip(td, Ym):
            pr.write_checkpoint(hdf, Ym_i, "concentration", t=ti)

    time = problem.timestepper.vector()
    M_mesh = matrix_operator(problem.model.M(coefficients))

    mesh_loss = LossFunction(td, Yd)
    mri_loss = MRILoss(evaluation_data=eval)
    M_mri = matrix_operator(mri_loss.M)

    sim_records = []
    for ti, Yi in zip(time, Y):
        sim_records.append(
            {
                "time": ti,
                "amount_mesh": M_mesh(Yi.vector()).sum() * 1e-6,
                "amount_mri": M_mri(Yi.vector()).sum() * 0.5**3 * 1e-6,
            }
        )
    pd.DataFrame.from_records(sim_records).to_csv(
        with_suffix(output, "_all.csv"),
        index=False,
    )

    time = problem.timestepper.vector()
    measure_records = []
    for idx, (ti, Ydi, Ymi) in enumerate(zip(td, Yd, Ym)):
        E = Ymi.vector() - Ydi.vector()
        e = M_mri(E)
        mri_sim_concentration = M_mri(Ymi.vector())
        measure_records.append(
            {
                "time": ti,
                "amount_mesh_sim": M_mesh(Ymi.vector()).sum(),
                "amount_mri_sim": mri_sim_concentration.sum() * 0.5**3 * 1e-6,
                "L2_squared_norm": mesh_loss.norms[idx],
                "L2_squared_error": mesh_loss._M_(E, E),
                "l2_squared_mri_error": e.inner(e),
            }
        )
    pd.DataFrame.from_records(measure_records).to_csv(
        with_suffix(output, "_measure.csv"), index=False
    )

    if visual:
        with df.XDMFFile(
            domain.mpi_comm(),
            str(with_suffix(output, "_measurements.xdmf")),
        ) as xdmf:
            for ti, Ym_i in zip(td, Ym):
                xdmf.write(Ym_i, t=ti)
        with df.XDMFFile(
            domain.mpi_comm(),
            str(with_suffix(output, "_all.xdmf")),
        ) as xdmf:
            for ti, Yi in zip(problem.timestepper.vector(), Y):
                xdmf.write(Yi, t=ti)


if __name__ == "__main__":
    main()
