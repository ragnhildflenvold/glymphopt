import click
import dolfin as df
import pandas as pd
import pantarei as pr
import numpy as np

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

    phi = coefficients["n_e"] + coefficients["n_p"]
    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)

    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))
    problem = MulticompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True
    )
    Y = problem.forward(coeffconverter.to_vector())
    Ym = problem.measure(Y)

    with df.HDF5File(domain.mpi_comm(), output, "w") as hdf:
        for ti, Ym_i in zip(td, Ym):
            pr.write_checkpoint(hdf, Ym_i, "concentration", t=ti)

    time = problem.timestepper.vector()

    M_mesh = matrix_operator(mass_matrix(Ym[0].function_space()))

    mesh_loss = LossFunction(td, Yd)
    mri_loss = MRILoss(evaluation_data=eval)
    M_mri = matrix_operator(mri_loss.M)

    sim_records = []
    Yi_vol = df.Function(Ym[0].function_space(), name="concentration")
    for ti, Yi in zip(time, Y):
        Yi_vol.assign(problem.volumetric_concentration(Yi))
        sim_records.append(
            {
                "time": ti,
                "amount_mesh": M_mesh(Yi_vol.vector()).sum() * 1e-6,  # cubic mm to l
                "amount_mri": M_mri(Yi_vol.vector()).sum()
                * 1e-6
                * 0.5**3,  # and voxel volume scaling
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

        mri_data_concentration = (
            np.array([np.nan]) if idx == 0 else mri_loss.Cd[idx - 1][:] * (3.2 / 4.5)
        )
        mri_sim_concentration = M_mri(Ymi.vector())
        measure_records.append(
            {
                "time": ti,
                "amount_mesh_sim": M_mesh(Ymi.vector()).sum() * 1e-6,
                "amount_mri_sim": mri_sim_concentration.sum() * 1e-6 * 0.5**3,
                "amount_mesh_data": M_mesh(Ydi.vector()).sum() * 1e-6,
                "amount_mri_data": mri_data_concentration.sum() * 1e-6 * 0.5**3,
                "L2_function_norm": mesh_loss.norms[idx],
                "L2_error": mesh_loss._M_(E, E),
                "l2_mri_data_norm": (mri_data_concentration**2).sum(),
                "l2_mri_error": e.inner(e),
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
