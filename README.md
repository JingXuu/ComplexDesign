# ComplexDesign

ComplexDesign is a constraint-aware hallucination framework for protein complex design.  
It supports **unconditional multichain generation** and **multi-target binder design**.  
ComplexDesign is built on the **BindCraft** and **ColabDesign** codebases.  

## Installation

Clone the repository:

```bash
git clone https://github.com/JingXuu/ComplexDesign.git
cd ComplexDesign
```

Run the installation script:

```bash
bash install_complexdesign.sh --cuda '12.4' --pkg_manager 'conda'
```

Before launching ComplexDesign in a new shell, activate the environment and export the runtime library path:

```bash
conda activate ComplexDesign
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

This export step is required in each new shell so that `DAlphaBall.gcc` can locate `libgfortran.so.5` at runtime.

## Running ComplexDesign

### Unconditional multichain generation

```bash
python complexdesign_unconditional.py \
  --settings ./settings_target/example_unconditional.json \
  --filters ./settings_filters/default_filters.json \
  --advanced ./settings_advanced/default_4stage_complexdesign_unconditional.json
```

### Multi-target binder design

```bash
python complexdesign_multitarget.py \
  --settings ./settings_target/example_multitarget.json \
  --filters ./settings_filters/default_filters.json \
  --advanced ./settings_advanced/default_4stage_complexdesign_multitarget.json
```

## Input requirements

For **multi-target binder design**, the input PDB should contain exactly two protein target chains.

For **unconditional multichain generation**, no input structure is required. Chain lengths are specified by `complex_lengths` in the target-setting JSON file.

## Configuration Files

* `settings_target/*.json`: defines the task, input files, output path, chain lengths or binder-length range, and the number of accepted designs to generate.
* `settings_filters/*.json`: defines confidence and structural filters used during and after optimization.
* `settings_advanced/*.json`: defines AF-M optimization settings, loss weights, and ProteinMPNN redesign settings.

## Outputs

Results are written under the `design_path` specified in the target-setting JSON file. Outputs include:

- `Trajectory/`: accepted designed structures, i.e. hallucination PDBs that pass the confidence and structure filters
- `MPNN/Sequences/`: ProteinMPNN-redesigned sequences for the accepted PDBs in `Trajectory/`
- `trajectory_stats.csv`: trajectories that pass the hallucination-stage confidence and structure filters
- `failure_csv.csv`: counts of failures

## Third-Party Dependencies

ComplexDesign is adapted from [BindCraft](https://github.com/martinpacesa/BindCraft), [ColabDesign](https://github.com/sokrypton/ColabDesign), and [ProteinMPNN](https://github.com/dauparas/ProteinMPNN). Please cite and follow the licenses of these projects where applicable.

[PyRosetta / RosettaCommons](https://www.pyrosetta.org/) is used by optional relaxation and interface-scoring utilities.

AlphaFold2 and AlphaFold-Multimer parameter files are downloaded during installation into `./params/`.

Additional third-party components retain their original licenses where applicable.

## Notes

- This public release provides the current design-stage implementation used in this work.
- The manuscript reports AF3-based downstream evaluation, but a complete AF3 evaluation stage is not included in the current public release.
