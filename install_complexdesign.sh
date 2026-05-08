#!/bin/bash
################## ComplexDesign installation script
# Adapted from the BindCraft installation script.
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'
cuda=''
env_name='ComplexDesign'

# Define the short and long options
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

### ComplexDesign install begin, create base environment
echo -e "Installing ${env_name} environment\n"
$pkg_manager create --name "${env_name}" python=3.10 -y || { echo -e "Error: Failed to create ${env_name} conda environment"; exit 1; }
conda env list | grep -w "${env_name}" >/dev/null 2>&1 || { echo -e "Error: Conda environment '${env_name}' does not exist after creation."; exit 1; }

# Load newly created ComplexDesign environment
echo -e "Loading ${env_name} environment\n"
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/${env_name}" || { echo -e "Error: Failed to activate the ${env_name} environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "${env_name}" ] || { echo -e "Error: The ${env_name} environment is not active."; exit 1; }
echo -e "${env_name} environment activated at ${CONDA_BASE}/envs/${env_name}"

# install required conda packages
echo -e "Installing conda requirements\n"
if [ -n "$cuda" ]; then
  CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
    -c conda-forge -c nvidia -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
else
  $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' \
    -c conda-forge -c nvidia -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
fi

# make sure all required packages were installed
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax)
if [ -n "$cuda" ]; then
    required_packages+=(cuda-nvcc cudnn)
fi
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# install local ColabDesign
echo -e "Installing local ColabDesign\n"
[ -d "${install_dir}/ColabDesign" ] || { echo -e "Error: Local ColabDesign directory not found at ${install_dir}/ColabDesign"; exit 1; }
(
  cd "${install_dir}/ColabDesign" &&
  pip install -e .
) || { echo -e "Error: Failed to install local ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# install PyRosetta
echo -e "Installing PyRosetta\n"
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization || { echo -e "Error: Failed to install PyRosetta"; exit 1; }
python -c "import pyrosetta" >/dev/null 2>&1 || { echo -e "Error: pyrosetta module not found after installation"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }

curl -L -C - \
  -o "${params_file}" \
  "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
  || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }

[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2 weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
conda deactivate
echo -e "${env_name} environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS
echo -e "Successfully finished ${env_name} installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate ${env_name}\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
