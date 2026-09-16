# glymphopt

Inverse parameter identification for gadolinium (MRI contrast) transport in brain tissue,
using single-compartment and two-compartment (ECS + PVS) diffusion-reaction PDE models solved
with FEniCS. For each subject, an adaptive grid search finds the model parameters that best
match simulated tracer concentrations to MRI-derived concentration estimates. This is the
pipeline behind the grid-search results (gamma/t_pb, alpha/r) reported in the two-compartment
gadolinium transport article.

## Setup

```bash
mkdir -p deps/
git clone -b boundary-cell-refinement \
  git@github.com:jorgenriseth/gMRI2FEM.git deps/gmri2fem
pixi install
```

To copy necessary input files from another source, without interfering with everyone elses work:
- Start by creating a `subjects.txt`-file listing the subjects of the study.
- Copy all necessary files from the source directory:
```bash
export SOURCEDIR=[full path to directory where mri_dataset and mri_processed_data is located]
./copy-deps.sh
```

If on a cluster with slurm execution, you need to create output directories first.
```bash
mkdir jobs logs
```
Pull the singularity container:
```bash
singularity build glymphopt.sif jorgenriseth/glymphopt
```
or build locally with the help of docker,
```bash
docker build -t glymphopt .;
apptainer build glymphopt.sif docker-daemon:glymphopt:latest
```

## Pipeline overview

Orchestrated per-subject via the `Snakefile` (subjects listed in `subjects.txt`):

1. `create_evaluation_data` — build a sparse matrix mapping mesh degrees of freedom to MRI
   voxel centers (`src/glymphopt/mri_loss.py`).
2. `gridsearch_singlecompartment` / `gridsearch_twocompartment` — adaptive grid search over
   the model parameters (`scripts/diffusion_reaction_gridsearch.py`,
   `scripts/twocomp_gridsearch.py`), producing a CSV of all evaluated points and their loss.
3. `run_optimal_singlecompartment` / `run_optimal_twocompartment` — re-run the forward model
   at the best-found parameters (`scripts/singlecompartment.py`, `scripts/twocompartment.py`).
4. `errortable` — compare simulated vs. measured concentrations region-by-region
   (`scripts/create_errortable.py`).

## Package modules (`src/glymphopt/`)

| Module | Purpose |
|---|---|
| `minimize.py` | Box-constrained Newton solver, plus the original `adaptive_grid_search` implementation. |
| `grid_scheduler.py`, `minimize_grid_search.py` | Refactored, stateless adaptive grid search (`GridScheduler` + `adaptive_grid_search`) used by the current grid-search scripts. |
| `measure.py` | Normalized L2 loss functions (`LossFunction`, `MRILoss`) comparing simulated and measured concentrations. |
| `mri_loss.py` | Builds the sparse mesh-to-MRI-voxel evaluation matrix used by `MRILoss`. |
| `twocompartment.py` | Two-compartment (ECS+PVS) forward/inverse PDE problem (`TwocompartmentModel`, `MulticompartmentInverseProblem`). |
| `singlecompartment.py` | Single-compartment equivalent forward/inverse PDE problem. |
| `operators.py` | FEniCS matrix assembly helpers (mass/stiffness/boundary/transfer matrices) and time-dependent coefficient base classes. |
| `parameters.py` | Default model parameters (volume fractions, diffusion, transfer coefficients, eta) with units via Pint. |
| `coefficientvectors.py` | Converts between named-parameter dicts and the flat vectors used by the optimizer. |
| `timestepper.py` | Uniform time grid + Crank-Nicolson bookkeeping. |
| `interpolation.py` | Linear interpolation of time-series data (boundary conditions, measurements). |
| `io.py` | HDF5/XDMF read/write, DTI tensor loading (`read_augmented_dti`). |
| `postprocessing.py` | Region-wise (parcellation) statistics for error tables. |
| `param_utils.py` | Encodes/parses parameter dicts to/from CLI- and filename-safe strings. |
| `scale.py` | Reduced-dimension wrapper for optimizing over a subspace of parameters. |
| `assigners.py` | Assigns/extracts fields between mixed function spaces and single compartments. |
| `cache.py` | Memoizes expensive forward/adjoint evaluations during optimization. |
| `differentiation.py`, `taylor.py` | Finite-difference gradient/Hessian checks (development/validation only). |
| `datageneration.py` | Weibull-shaped synthetic boundary concentration, used for 2D test cases. |
| `blood.py` | Blood plasma concentration models (`BatemanModel`, `BiexponentialModel`). **Currently unused/dead code** — not imported anywhere else in the package; blood-curve fitting for the article was done outside this repo. |
| `utils.py` | Misc helpers (path suffixing, dict flattening). |

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `diffusion_reaction_gridsearch.py`, `twocomp_gridsearch.py` | Core adaptive grid search per subject; outputs the parameter/loss CSVs. |
| `singlecompartment.py`, `twocompartment.py` | Forward solve at given (typically optimal) parameters. |
| `singlecompartment_convergence.py`, `twocompartment_convergence.py` | Numerical convergence study (varying timestep/mesh resolution), tracking total tracer amount and measurement-time error — produces the article's convergence figure. |
| `singlecompartment_eval_point.py`, `twocompartment_eval_point.py` | Evaluate the loss at a single given parameter point (used for cluster-parallelized grid search). |
| `collect_grid_value.py`, `collect_mesh_data.py` | Collect per-point evaluations / per-subject mesh+data bundles produced by cluster jobs into a single file. |
| `voxel_center_minimization.py` | Builds the mesh-to-MRI-voxel evaluation matrix and reconstructs mesh functions from MRI voxel data (`create_evaluation_matrix`, `map_mri_to_mesh`). |
| `extract_timestamps.py` | Reads acquisition timestamps for a subject/sequence from the study timetable (via the external `gmri2fem` package). |
| `create_errortable.py` | Region-wise comparison table between data, single-, and two-compartment results. |
| `diffusion_reaction_minimization.py` | Alternative L-BFGS-B-based optimization; incomplete/exploratory, not used in the Snakemake pipeline. |
| `datageneration2d.py` | Generates synthetic 2D test data; not part of the subject pipeline. |
| `test_script_evaluate.py` | Toy loss function (distance to a fixed optimum) for testing the grid-search/cluster-scheduling machinery. |

## Notebooks & sandbox

`notebooks/` (`data-generation.ipynb`, `inverse-model.ipynb`, `inverse-multicompartment.ipynb`)
are exploratory/development notebooks used to prototype the inverse problems; the production
path is the scripts above. `sandbox/dolfin-adjoint.py` is an old dolfin-adjoint-based
experiment, not used elsewhere.

## Cluster execution

`profile/config.yaml` configures Snakemake for Slurm execution via Singularity
(`grid_scheduler.py`/`*_eval_point.py` scripts support splitting one grid-search iteration across
cluster jobs). `Dockerfile`, `copy-deps.sh`, `depfiles.txt`, and `snakebatch.sh` support building
the container and staging input data for cluster runs.

## Relationship to `threecomp`

The sibling `threecomp` repository implements a related but distinct forward-solve framework
(single- and three-compartment models with fixed parameter sweeps, no adaptive optimization).
The two repositories independently reimplement similar utility layers (matrix assembly, I/O,
time-stepping, interpolation) rather than sharing code. `threecomp` is not the source of this
article's grid-search results.

## Known gaps (not implemented in this repo)

Note the merged branch was named `mri-measurement-error`, but that refers to the MRI-based
loss/error machinery (`mri_loss.py`'s voxel-evaluation matrix, the `*_convergence.py` scripts
comparing simulated vs. MRI error) — which *is* now present here. It is **not** the same as the
article's "Data noise estimates" section. The following are still not implemented in this repo:
the blood-concentration curve fitting (see `blood.py` note above), the eta boundary-scaling
regression, and the white-matter-ROI image-noise estimation. Mesh generation, DTI processing,
and MRI-based concentration estimation are also external, and live in the upstream
preprocessing repository referenced in Setup above, `gMRI2FEM` (branch
`boundary-cell-refinement`).

