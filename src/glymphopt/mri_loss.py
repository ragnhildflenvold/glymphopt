import click
import pantarei as pr
import numpy as np
import dolfin as df
import scipy as sp
import simple_mri as sm
import tqdm

from petsc4py import PETSc


def sparse_matrix_to_dolfin(A):
    A_petsc = sparse_scipy_to_petsc(A)
    return df.Matrix(df.PETScMatrix(A_petsc))


def petsc_to_scipy(petsc_mat):
    m, n = petsc_mat.getSize()
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.sparse.csr_matrix((data, indices, indptr), shape=(m, n))


def sparse_scipy_to_petsc(A):
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


def numpy_array_to_dolfin_vector(v):
    vec = df.Vector(df.MPI.comm_world, len(v))
    vec[:] = v
    return vec


def numpy_array_to_petsc_vector(v):
    petsc_vec = df.PETScVector(df.MPI.comm_world)
    petsc_vec.init(len(v))
    petsc_vec.set_local(v)


def create_evaluation_matrix(
    V,
    data_mris,
):
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
    M = sp.sparse.lil_matrix((len(xyz), V.dim()))
    for i, (xi, cell_index_i) in enumerate(zip(tqdm.tqdm(xyz), cells_containing_point)):
        cell_global_dofs = V.dofmap().cell_dofs(cell_index_i)
        cell = df.Cell(V.mesh(), cell_index_i)
        dof_vals = V.element().evaluate_basis_all(
            xi, cell.get_vertex_coordinates(), cell.orientation()
        )
        M[i, cell_global_dofs] = dof_vals
    return M.tocsr(), ijk


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.argument("mris", type=str, nargs=-1)
def main(
    input,
    output,
    mris,
):
    with df.HDF5File(df.MPI.comm_world, input, "r") as hdf:
        mesh = pr.read_domain(hdf)
        V = df.FunctionSpace(mesh, "CG", 1)
    data_mris = [sm.load_mri(p, dtype=np.single) for p in mris]
    M, ijk = create_evaluation_matrix(V, data_mris)
    np.savez_compressed(
        output,
        matrix_data=M.data,
        matrix_indices=M.indices,
        matrix_indptr=M.indptr,
        matrix_shape=M.shape,
        mri_indices=ijk,
        **{f"vector{idx}": mri.data[*ijk.T] for idx, mri in enumerate(data_mris)},
    )


if __name__ == "__main__":
    main()
