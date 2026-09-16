########################
## Setup
########################
container: "glymphopt.sif"

if DeploymentMethod.APPTAINER in workflow.deployment_settings.deployment_method:
  shell.prefix(
    "set -eo pipefail; "
    "/app/entrypoint.sh "
  )

with open("subjects.txt", "r") as f:
    SUBJECTS = [line.strip() for line in f.readlines()]

def get_optimal(file, key):
  import pandas as pd
  dframe = pd.read_csv(file, sep=";")
  return float(dframe.loc[dframe["funceval"].idxmin()][key])

def looklocker_exists(subject, session):
  looklocker = f"mri_dataset/{subject}/{session}/anat/{subject}_{session}_acq-looklocker_IRT1.nii.gz"
  return Path(looklocker).exists()

def mixed_exists(subject, session):
  mixed = f"mri_dataset/{subject}/{session}/mixed/{subject}_{session}_acq-mixed_SE-modulus.nii.gz"
  return Path(mixed).exists()

def concentration_exists(subject, session):
  concentration = f"mri_processed_data/{subject}/concentrations/{subject}_{session}_concentration.nii.gz"
  return Path(concentration).exists()



###################################
## Preprocessing
###################################
rule extract_timestamps:
  input:
    "mri_dataset/timetable.tsv"
  output:
    "mri_processed_data/{subject}/modeling/timestamps.txt"
  params:
    sequence="looklocker"
  shell:
    "python scripts/extract_timestamps.py"
    " -t {input}"
    " -o {output}"
    " -sub {wildcards.subject}"
    " -seq {params.sequence}"


rule generate_mesh:
  input:
    "mri_processed_data/{subject}/modeling/surfaces/lh_pial_refined.stl",
    "mri_processed_data/{subject}/modeling/surfaces/rh_pial_refined.stl",
    "mri_processed_data/{subject}/modeling/surfaces/subcortical_gm.stl",
    "mri_processed_data/{subject}/modeling/surfaces/ventricles.stl",
    "mri_processed_data/{subject}/modeling/surfaces/white.stl",
  output:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/mesh-raw.hdf",
    xdmf="mri_processed_data/{subject}/modeling/resolution{res}/mesh_xdmfs/subdomains.xdmf"
  shell:
    "gmri2fem brainmeshing meshgen"
    " --surface_dir $(dirname {input[0]})"
    " --resolution {wildcards.res}"
    " --output {output.hdf}"


rule refine_mesh_boundary:
  input:
    "mri_processed_data/{subject}/modeling/resolution{res}/mesh-raw.hdf",
  output:
    "mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
  params:
    cellsize=1.0,
    maxiter=1,
  shell:
    "gmri2fem brainmeshing refine_mesh"
    " -i {input} -o {output} -t {params.cellsize} --maxiter {params.maxiter}"


rule create_evaluation_data:
  input:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
    concentrations= lambda wc: [
      f"mri_processed_data/{wc.subject}/concentrations/{wc.subject}_{ses}_concentration.nii.gz"
      for ses in (f"ses-{idx:02d}" for idx in range(1, 6))
      if concentration_exists(wc.subject, ses)
    ],
    csfmask = "mri_processed_data/{subject}/segmentations/{subject}_seg-csf_binary.nii.gz",
    timestamps="mri_processed_data/{subject}/modeling/timestamps.txt",
  params: 
    concentrations=lambda wc, input: " ".join(input.concentrations)
  output:
    "mri_processed_data/{subject}/modeling/resolution{res}/evaluation_data.npz"
  shell:
    "gmri2fem i2m evaluation_data"
    " {params.concentrations}"
    " --input {input.hdf}"
    " --output {output}"
    " --csfmask {input.csfmask}"
    " --timestamps {input.timestamps}"


rule mesh_segmentation:
  input:
    seg="mri_processed_data/fastsurfer/{subject}/mri/{seg}.mgz",
    mesh="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
  output:
    "mri_processed_data/{subject}/modeling/resolution{res}/mesh_seg-{seg}.hdf"
  shell:
    "gmri2fem i2m subdomains"
    " {input} {output}"

rule dti2fenics:
  input:
    meshfile="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
    dti="mri_processed_data/{subject}/dwi/{subject}_ses-01_dDTI_cleaned.nii.gz",
    md = "mri_processed_data/{subject}/registered/{subject}_ses-01_dDTI_MD_registered.nii.gz",
    fa = "mri_processed_data/{subject}/registered/{subject}_ses-01_dDTI_FA_registered.nii.gz",
    mask= "mri_processed_data/{subject}/segmentations/{subject}_seg-aseg_refined.nii.gz",
  output:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/dti.hdf",
  shell:
    "gmri2fem i2m dti2mesh"
    " --mesh {input.meshfile}"
    " --dti {input.dti}"
    " --md {input.md}"
    " --fa {input.fa}"
    " --mask {input.mask}" # functions as brainmask
    " --output {output.hdf}"

rule collect:
  input:
    mesh="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
    tissue="mri_processed_data/{subject}/modeling/resolution{res}/tissue_concentrations.hdf",
    boundary="mri_processed_data/{subject}/modeling/resolution{res}/boundary_concentrations.hdf",
    dti="mri_processed_data/{subject}/modeling/resolution{res}/dti.hdf",
    parcellations="mri_processed_data/{subject}/modeling/resolution{res}/mesh_seg-wmparc.hdf"
  output:
    "mri_processed_data/{subject}/modeling/resolution{res}/data.hdf"
  shell:
    "python scripts/collect_mesh_data.py"
    " --domain {input.mesh}"
    " --dti_data {input.dti}"
    " --tissue_concentrations {input.tissue}"
    " --boundary_concentrations {input.boundary}"
    " --parcellation_data {input.parcellations}"
    " --output {output}"


rule boundary_concentrations:
  input:
    mesh="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
    csfmask="mri_processed_data/{subject}/segmentations/{subject}_seg-csf_binary.nii.gz",
    timestamps="mri_processed_data/{subject}/modeling/timestamps.txt",
    concentrations= lambda wc: [
      f"mri_processed_data/{wc.subject}/concentrations/{wc.subject}_{ses}_concentration.nii.gz"
      for ses in (f"ses-{idx:02d}" for idx in range(1, 6))
      if concentration_exists(wc.subject, ses)
    ],
  output:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/boundary_concentrations.hdf",
  params: 
    concentrations=lambda wc, input: " ".join(input.concentrations)
  shell:
    "gmri2fem i2m boundary-concentrations"
    " -m {input.mesh}"
    " -o {output}"
    " -t {input.timestamps}"
    " -c {input.csfmask}"
    " {params.concentrations}"


rule tissue_concentrations:
  input:
    mesh="mri_processed_data/{subject}/modeling/resolution{res}/mesh.hdf",
    eval="mri_processed_data/{subject}/modeling/resolution{res}/evaluation_data.npz",
  output:
    "mri_processed_data/{subject}/modeling/resolution{res}/tissue_concentrations.hdf",
  shell:
    "gmri2fem i2m tissue-concentrations"
    " -m {input.mesh}"
    " -e {input.eval}"
    " -o {output}"
    " --verbose"
    " --visual"



rule gridsearch_twocompartment:
    input:
        "mri_processed_data/{subject}/modeling/resolution32/data.hdf"
    output:
        "results/{subject}_twocomp_gridsearch.csv"
    params:
      iterations = 1
    resources:
      runtime="30h"
    shell:
        "python scripts/twocomp_gridsearch.py"
        " -i {input}"
        " -o {output}"
        " --iter {params.iterations}"


rule run_optimal_singlecompartment:
  input:
    "results/{subject}_singlecomp_gridsearch.csv",
    data="mri_processed_data/{subject}/modeling/resolution32/data.hdf",
  output:
    "results/{subject}_singlecomp_optimal.hdf",
  params:
    alpha = lambda wc: get_optimal(f"results/{wc.subject}_singlecomp_gridsearch.csv", "a"),
    r = lambda wc: get_optimal(f"results/{wc.subject}_singlecomp_gridsearch.csv", "r")
  shell:
    "python scripts/singlecompartment.py"
    " -i {input.data}"
    " -o {output}"
    " -a {params.alpha}"
    " -r {params.r}"

rule run_optimal_twocompartment:
  input:
    "results/{subject}_twocomp_gridsearch.csv",
    data="mri_processed_data/{subject}/modeling/resolution32/data.hdf",
  output:
    "results/{subject}_twocomp_optimal.hdf",
  params:
    gamma = lambda wc: get_optimal(f"results/{wc.subject}_twocomp_gridsearch.csv", "gamma"),
    t_pb = lambda wc: get_optimal(f"results/{wc.subject}_twocomp_gridsearch.csv", "t_pb")
  shell:
    "python scripts/twocompartment.py"
    " -i {input.data}"
    " -o {output}"
    " --gamma {params.gamma}"
    " --t_pb {params.t_pb}"

rule errortable:
  input:
    data="mri_processed_data/{subject}/modeling/resolution32/data.hdf",
    single="results/{subject}_singlecomp_optimal.hdf",
    multi="results/{subject}_twocomp_optimal.hdf",
  output:
    "results/{subject}_errortable.csv"
  shell:
    "python scripts/create_errortable.py"
    " --datapath {input.data}"
    " --singlecomp {input.single}"
    " --twocomp {input.multi}"
    " --output {output}"


from glymphopt.param_utils import float_string_formatter


rule singlecomp_converge_analysis_workflow:
  input:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/data.hdf",
    eval="mri_processed_data/{subject}/modeling/resolution{res}/evaluation_data.npz"
  output:
    "results/singlecomp_convergence/{subject}_resolution-{res}_dt-{dt}_alpha-{alpha}_r-{r}.hdf"
  params:
    alpha = lambda wc: float(wc.alpha),
    r = lambda wc: float(wc.r)
  shell:
    "python scripts/singlecompartment_convergence.py"
    " -i {input.hdf}"
    " -o {output}"
    " -e {input.eval}"
    " --dt {wildcards.dt}"
    " --a {params.alpha}"
    " --r {params.r}"


rule twocomp_convergence_analysis_workflow:
  input:
    hdf="mri_processed_data/{subject}/modeling/resolution{res}/data.hdf",
    eval="mri_processed_data/{subject}/modeling/resolution{res}/evaluation_data.npz"
  output:
    "results/twocomp_convergence/{subject}_resolution-{res}_dt-{dt}_gamma-{gamma}_t_pb-{t_pb}.hdf"
  params:
    gamma = lambda wc: float(wc.gamma),
    t_pb = lambda wc: float(wc.t_pb)
  shell:
    "python scripts/twocompartment_convergence.py"
    " -i {input.hdf}"
    " -o {output}"
    " -e {input.eval}"
    " --dt {wildcards.dt}"
    " --gamma {params.gamma}"
    " --t_pb {params.t_pb}"



#####################
# Adaptive Gridsearch Singlecompartment
#####################
localrules: define_initial_grid, refine_grid, collect_initial_grid, collect_grid

def read_grid(file):
  try:
    with open(file, "r") as f:
      filenames = f.read().split()
  except:
    filenames = []
  return filenames

ALPHA_BOUNDS = (1, 5)
R_BOUNDS = (0, 5e-5)

ruleorder: define_initial_grid > refine_grid
checkpoint define_initial_grid:
  input:
    "mri_processed_data/{subject}/modeling/resolution30/data.hdf",
    "mri_processed_data/{subject}/modeling/resolution30/evaluation_data.npz",
    "mri_processed_data/{subject}/modeling/timestamps.txt"
  output:
    "results/singlecompartment_gridsearch/{subject}/grid0.csv"
  params:
    alpha_bounds = lambda wc: " ".join(map(str, ALPHA_BOUNDS)),
    r_bounds = lambda wc: " ".join(map(str, R_BOUNDS)),
  shell:
    "python scripts/grid_scheduler.py"
    " --param alpha {params.alpha_bounds}"
    " --param r {params.r_bounds}"
    " -o {output}"


rule gridsearch_singlecompartment:
    input:
        "mri_processed_data/{subject}/modeling/resolution32/data.hdf"
    output:
        "results/{subject}_singlecomp_gridsearch.csv"
    shell:
        "python scripts/diffusion_reaction_gridsearch.py"
        " -i {input}"
        " -o {output}"

rule singlecompartment_point_evaluation:
  input:
    hdf="mri_processed_data/{subject}/modeling/resolution30/data.hdf",
    eval="mri_processed_data/{subject}/modeling/resolution30/evaluation_data.npz"
  output:
    "results/singlecompartment_gridsearch/{subject}/alpha{alpha}_r{r}_error.json"
  params:
    a = lambda wc: float(wc.alpha),
    r = lambda wc: float(wc.r)
  threads: 1
  resources:
      cpus_per_task=1,
      runtime="20m"
  shell:
    "python scripts/singlecompartment_eval_point.py"
    " -i {input.hdf}"
    " -e {input.eval}"
    " -o {output}"
    " --a {params.a}"
    " --r {params.r}"

ruleorder: collect_initial_grid > collect_grid
rule collect_initial_grid:
  input:
    "results/singlecompartment_gridsearch/{subject}/grid0.csv",
    sim_results = lambda wc: [
      f"results/singlecompartment_gridsearch/{wc.subject}/{file}"
      for file in read_grid(f"results/singlecompartment_gridsearch/{wc.subject}/grid0.csv")
    ]
  output:
    "results/singlecompartment_gridsearch/{subject}/history0.csv"
  params:
    input_list = lambda wc, input: " ".join(input.sim_results)
  shell:
    "python scripts/collect_grid_value.py"
    " {params.input_list}"
    " -o {output}"

checkpoint refine_grid:
  input:
    lambda wc: f"results/singlecompartment_gridsearch/{wc.subject}/history{int(wc.iter)-1}.csv"
  output:
    "results/singlecompartment_gridsearch/{subject}/grid{iter}.csv"
  params:
    alpha_bounds = lambda wc: " ".join(map(str,ALPHA_BOUNDS)),
    r_bounds = lambda wc: " ".join(map(str, R_BOUNDS)),
  shell:
    "python scripts/grid_scheduler.py"
    " --param alpha {params.alpha_bounds}"
    " --param r {params.r_bounds}"
    " --history {input}"
    " -o {output}"

rule collect_grid:
  input:
    grid="results/singlecompartment_gridsearch/{subject}/grid{iter}.csv",
    history=lambda wc: f"results/singlecompartment_gridsearch/{wc.subject}/history{int(wc.iter)-1}.csv",
    sim_results = lambda wc: [f"results/singlecompartment_gridsearch/{wc.subject}/{file}"
      for file in read_grid(f"results/singlecompartment_gridsearch/{wc.subject}/grid{wc.iter}.csv")
    ]
  output:
    "results/singlecompartment_gridsearch/{subject}/history{iter}.csv"
  params:
    input_list = lambda wc, input: " ".join(input.sim_results)
  shell:
    "python scripts/collect_grid_value.py"
    " {params.input_list}"
    " -o {output}"
    " --history {input.history}"

##############################
### Adaptive twocomp gridsearch
#############################
localrules: define_initial_grid, refine_grid, collect_initial_grid, collect_grid

GAMMA_BOUNDS = (1, 45)
T_PB_BOUNDS = (0, 1e-5)

ruleorder: define_initial_grid_twocompartment > refine_grid_twocompartment
checkpoint define_initial_grid_twocompartment:
  input:
    "mri_processed_data/{subject}/modeling/resolution30/data.hdf",
    "mri_processed_data/{subject}/modeling/resolution30/evaluation_data.npz",
    "mri_processed_data/{subject}/modeling/timestamps.txt"
  output:
    "results/twocompartment_gridsearch/{subject}/grid0.csv"
  params:
    gamma_bounds = lambda wc: " ".join(map(str, GAMMA_BOUNDS)),
    t_pb_bounds = lambda wc: " ".join(map(str, T_PB_BOUNDS)),
  shell:
    "python scripts/grid_scheduler.py"
    " --param gamma {params.gamma_bounds}"
    " --param t_pb {params.t_pb_bounds}"
    " -o {output}"

rule twocompartment_point_evaluation:
  input:
    hdf="mri_processed_data/{subject}/modeling/resolution30/data.hdf",
    eval="mri_processed_data/{subject}/modeling/resolution30/evaluation_data.npz"
  output:
    "results/twocompartment_gridsearch/{subject}/gamma{gamma}_t_pb{t_pb}_error.json"
  params:
    gamma = lambda wc: float(wc.gamma),
    t_pb = lambda wc: float(wc.t_pb)
  threads: 1
  resources:
      cpus_per_task=1,
  shell:
    "python scripts/twocompartment_eval_point.py"
    " -i {input.hdf}"
    " -e {input.eval}"
    " -o {output}"
    " --gamma {params.gamma}"
    " --t_pb {params.t_pb}"

ruleorder: collect_initial_grid_twocompartment > collect_grid_twocompartment
rule collect_initial_grid_twocompartment:
  input:
    "results/twocompartment_gridsearch/{subject}/grid0.csv",
    sim_results = lambda wc: [
      f"results/twocompartment_gridsearch/{wc.subject}/{file}"
      for file in read_grid(f"results/twocompartment_gridsearch/{wc.subject}/grid0.csv")
    ]
  output:
    "results/twocompartment_gridsearch/{subject}/history0.csv"
  params:
    input_list = lambda wc, input: " ".join(input.sim_results)
  shell:
    "python scripts/collect_grid_value.py"
    " {params.input_list}"
    " -o {output}"

checkpoint refine_grid_twocompartment:
  input:
    lambda wc: f"results/twocompartment_gridsearch/{wc.subject}/history{int(wc.iter)-1}.csv"
  output:
    "results/twocompartment_gridsearch/{subject}/grid{iter}.csv"
  params:
    gamma_bounds = lambda wc: " ".join(map(str,GAMMA_BOUNDS)),
    t_pb_bounds = lambda wc: " ".join(map(str, T_PB_BOUNDS)),
  shell:
    "python scripts/grid_scheduler.py"
    " --param gamma {params.gamma_bounds}"
    " --param t_pb {params.t_pb_bounds}"
    " --history {input}"
    " -o {output}"

rule collect_grid_twocompartment:
  input:
    grid="results/twocompartment_gridsearch/{subject}/grid{iter}.csv",
    history=lambda wc: f"results/twocompartment_gridsearch/{wc.subject}/history{int(wc.iter)-1}.csv",
    sim_results = lambda wc: [f"results/twocompartment_gridsearch/{wc.subject}/{file}"
      for file in read_grid(f"results/twocompartment_gridsearch/{wc.subject}/grid{wc.iter}.csv")
    ]
  output:
    "results/twocompartment_gridsearch/{subject}/history{iter}.csv"
  params:
    input_list = lambda wc, input: " ".join(input.sim_results)
  shell:
    "python scripts/collect_grid_value.py"
    " {params.input_list}"
    " -o {output}"
    " --history {input.history}"
