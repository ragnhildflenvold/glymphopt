import click
import panta_rhei as pr
import simple_mri as sm
import dolfin as df
import numpy as np
import scipy
import tqdm
from dolfin import inner, grad
from glymphopt.utils import with_suffix


def create_evaluation_matrix(
    V: df.FunctionSpace,
    data_mris: list[sm.SimpleMRI],
):
    """
    ***NB:*** Although tempting to split this into one function for matrix creation,
    and one function for determination of valid voxel indices, this function
    needs to be one, since there is some memory bug occuring once some not
    yet known variable goes out of scope.
    """
    mesh = V.mesh()
    affine = data_mris[0].affine
    mask = np.prod([np.isfinite(mri.data) for mri in data_mris], axis=0)
    IJK = np.array(np.where(mask)).T
    XYZ = sm.apply_affine(affine, IJK)

    tree = mesh.bounding_box_tree()
    cells_containing_point = np.array(
        [tree.compute_first_entity_collision(df.Point(*xi)) for xi in tqdm.tqdm(XYZ)]
    )
    (points_of_interest,) = np.where(cells_containing_point < mesh.num_cells())
    cells_containing_point = cells_containing_point[points_of_interest]
    ijk = IJK[points_of_interest]
    xyz = XYZ[points_of_interest]
    M = scipy.sparse.lil_matrix((len(xyz), V.dim()))
    for i, (xi, cell_index_i) in enumerate(zip(tqdm.tqdm(xyz), cells_containing_point)):
        cell_global_dofs = V.dofmap().cell_dofs(cell_index_i)
        cell = df.Cell(V.mesh(), cell_index_i)
        dof_vals = V.element().evaluate_basis_all(
            xi, cell.get_vertex_coordinates(), cell.orientation()
        )
        M[i, cell_global_dofs] = dof_vals
    return M.tocsr(), ijk


@click.command()
@click.argument("mris", type=str, nargs=-1)
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--csf_mask", type=str)
def create_mesh_evaluation_data(mris, input, output, csf_mask=""):
    with df.HDF5File(df.MPI.comm_world, input, "r") as hdf:
        mesh = pr.read_domain(hdf)
    V = df.FunctionSpace(mesh, "CG", 1)
    data_mris = [sm.load_mri(p, dtype=np.single) for p in mris]
    if csf_mask:
        mask_mri = sm.load_mri(csf_mask, dtype=bool)
        for mri in data_mris:
            mri.data[mask_mri.data] = np.nan

    M, ijk = create_evaluation_matrix(V, data_mris)
    np.savez_compressed(
        output,
        matrix_data=M.data,
        matrix_indices=M.indices,
        matrix_indptr=M.indptr,
        matrix_shape=M.shape,
        mri_indices=ijk,
        **{f"vector{idx}": mri.data[*ijk.T] for idx, mri in enumerate(data_mris)},  # type: ignore
        fem_family="CG",
        fem_degree=1,
    )


def petsc_to_scipy(petsc_mat):
    m, n = petsc_mat.getSize()
    indptr, indices, data = petsc_mat.getValuesCSR()
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))


def construct_minimal_evaluation_system(V, M):
    (nonzero_cols,) = np.where(M.getnnz(0) != 0)
    A = M.T @ M
    A = A[nonzero_cols].tocsr()

    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = df.Measure("dx", domain=V.mesh())
    K = df.assemble(inner(grad(u), grad(v)) * dx)

    K_sp = petsc_to_scipy(df.as_backend_type(K).mat())
    AA = scipy.sparse.bmat(
        [[K_sp, A.T], [A, scipy.sparse.csr_matrix((A.shape[0], A.shape[0]))]]
    ).tocsr()
    return AA


def construct_minimal_evaluation_vector(M, x):
    (nonzero_cols,) = np.where(M.getnnz(0) != 0)
    b = M.T @ x
    bb = np.zeros(len(b) + len(nonzero_cols))
    bb[len(b) :] = b[nonzero_cols]
    return bb


def map_mri_to_mesh(
    C: list[np.ndarray], V: df.FunctionSpace, M: scipy.sparse.csr_matrix, verbose=True
) -> list[df.Function]:
    A = construct_minimal_evaluation_system(V, M)
    B = [construct_minimal_evaluation_vector(M, c) for c in C]
    U = [df.Function(V, name="concentration") for _ in range(len(C))]
    for idx, b in enumerate(tqdm.tqdm(B, disable=(not verbose))):
        uu, returncode = scipy.sparse.linalg.minres(A, b, show=verbose)
        U[idx].vector()[:] = uu[: V.dim()]
    return U


@click.command()
@click.option("--mesh", "-m", type=str, required=True)
@click.option("--eval", "-e", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--verbose", type=bool, is_flag=True)
@click.option("--visual", type=bool, is_flag=True)
def map_evaluation_data_to_mesh(
    mesh_path: str,
    evaluation_path: str,
    output: str,
    verbose: bool = True,
    visual: bool = True,
):
    with df.HDF5File(df.MPI.comm_world, mesh_path, "r") as hdf:
        mesh = pr.read_domain(hdf)

    npzfile = np.load(evaluation_path)
    M = scipy.sparse.csr_matrix(
        (
            npzfile["matrix_data"],
            npzfile["matrix_indices"],
            npzfile["matrix_indptr"],
        ),
        shape=npzfile["matrix_shape"],
    )
    C = [npzfile[f"vector{idx}"] for idx in range(5)]
    family, degree = str(npzfile["fem_family"]), int(npzfile["fem_degree"])

    V = df.FunctionSpace(mesh, family, degree)
    U = map_mri_to_mesh(C, V, M, verbose)

    with df.HDF5File(df.MPI.comm_world, output, "r") as hdf:
        pr.write_domain(hdf, mesh)
        u = U[0]
        pr.write_function(hdf, u, str(u.name))
        for i, u in enumerate(U[1:], start=1):
            pr.write_checkpoint(hdf, u, str(u.name), t=float(i))

    if visual:
        with df.XDMFFile(df.MPI.comm_world, with_suffix(output, ".xdmf")) as xdmf:
            for i, u in enumerate(U):
                xdmf.write(u, t=i)


if __name__ == "__main__":
    create_mesh_evaluation_data()
