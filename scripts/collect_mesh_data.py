from pathlib import Path
from typing import Optional

import click
import dolfin as df
import panta_rhei as pr


def collect_mesh_data(
    domain_data: Path,
    tissue_concentrations: Path,
    boundary_concentrations: Path,
    parcellation_data: Path,
    output: Path,
    dti_data: Optional[Path] = None,
):
    hdf = df.HDF5File(df.MPI.comm_world, str(output), "w")
    domain_hdf = df.HDF5File(df.MPI.comm_world, str(domain_data), "r")
    domain = pr.read_domain(domain_hdf)
    pr.write_domain(hdf, domain)
    pr.close(domain_hdf)

    boundary_hdf = df.HDF5File(df.MPI.comm_world, str(boundary_concentrations), "r")
    func = pr.read_function(boundary_hdf, "boundary_concentration", domain)
    time = pr.read_timevector(boundary_hdf, "boundary_concentration")
    pr.write_function(hdf, func, "boundary_concentration")
    for idx, ti in enumerate(time[1:], start=1):
        pr.read_checkpoint(boundary_hdf, func, "boundary_concentration", idx)
        pr.write_checkpoint(hdf, func, "boundary_concentration", ti)
    pr.close(boundary_hdf)

    tissue_hdf = df.HDF5File(df.MPI.comm_world, str(tissue_concentrations), "r")
    func = pr.read_function(tissue_hdf, "concentration", domain)
    time = pr.read_timevector(tissue_hdf, "concentration")
    pr.write_function(hdf, func, "concentration")
    for idx, ti in enumerate(time[1:], start=1):
        pr.read_checkpoint(tissue_hdf, func, "concentration", idx)
        pr.write_checkpoint(hdf, func, "concentration", ti)
    pr.close(tissue_hdf)

    if dti_data is not None:
        dti_hdf = df.HDF5File(df.MPI.comm_world, str(dti_data), "r")
        for funcname in ["DTI", "MD"]:
            func = pr.read_function(dti_hdf, funcname, domain)
            pr.write_function(hdf, func, funcname)
        pr.close(dti_hdf)

    parcellation_hdf = df.HDF5File(df.MPI.comm_world, str(parcellation_data), "r")
    subdomains = df.MeshFunction("size_t", domain, domain.topology().dim())
    parcellation_hdf.read(subdomains, "parcellations")
    hdf.write(subdomains, "parcellations")
    hdf.close()


@click.command()
@click.option("--domain", "domain_data", type=Path, required=True)
@click.option("--tissue_concentrations", type=Path, required=True)
@click.option("--boundary_concentrations", type=Path, required=True)
@click.option("--parcellation_data", type=Path, required=True)
@click.option("--output", type=Path, required=True)
@click.option("--dti_data", type=Path)
def collect(*args, **kwargs):
    collect_mesh_data(*args, **kwargs)


if __name__ == "__main__":
    collect()
