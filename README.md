# glymphopt

Inverse parameter identification for gadolinium (MRI contrast) transport in brain tissue,
using single-compartment and two-compartment (ECS + PVS) diffusion-reaction PDE models solved
with FEniCS. For each subject, an adaptive grid search finds the model parameters that best
match simulated tracer concentrations to MRI-derived concentration estimates. This is the
pipeline behind the grid-search results (gamma/t_pb, alpha/r) reported in the two-compartment
gadolinium transport article.

> **Note:** the locally checked-out `main` branch appears to lag behind the remote
> `mri-measurement-error` branch (10 commits ahead, not checked out here), which contains
> additional convergence-study scripts (`singlecompartment_convergence.py`,
> `twocompartment_convergence.py`) and a default `eta=0.39` matching the article exactly
> (vs. `eta=0.4` hardcoded in this branch's scripts). Check which branch/commit produced the
> final article numbers before relying on `main` alone.

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
| `minimize.py` | Adaptive grid search (`adaptive_grid_search`) and a box-constrained Newton solver — the core optimization algorithm. |
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
| `create_errortable.py` | Region-wise comparison table between data, single-, and two-compartment results. |
| `diffusion_reaction_minimization.py` | Alternative L-BFGS-B-based optimization; incomplete/exploratory, not used in the Snakemake pipeline. |
| `datageneration2d.py` | Generates synthetic 2D test data; not part of the subject pipeline. |

## Notebooks & sandbox

`notebooks/` (`data-generation.ipynb`, `inverse-model.ipynb`, `inverse-multicompartment.ipynb`)
are exploratory/development notebooks used to prototype the inverse problems; the production
path is the scripts above. `sandbox/dolfin-adjoint.py` is an old dolfin-adjoint-based
experiment, not used elsewhere.

## Relationship to `threecomp`

The sibling `threecomp` repository implements a related but distinct forward-solve framework
(single- and three-compartment models with fixed parameter sweeps, no adaptive optimization).
The two repositories independently reimplement similar utility layers (matrix assembly, I/O,
time-stepping, interpolation) rather than sharing code. `threecomp` is not the source of this
article's grid-search results.

## Known gaps (not implemented in this repo)

The article's blood-concentration curve fitting, the eta boundary-scaling regression, and the
MRI noise estimation are not implemented here (see `blood.py` note above). Mesh generation,
DTI processing, and MRI-based concentration estimation are also external. These likely live in
the upstream preprocessing repository referenced by this project's setup instructions,
`gMRI2FEM` (branch `boundary-cell-refinement`).
