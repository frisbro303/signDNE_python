# Signed ariaDNE CLI

This tool calculates the Signed DNE for 3d meshes. It provides options for visualization and customization of the calculation parameters.
# Index

## Index
1. [Introduction](#introduction)
2. [Install](#install)
3. [Command line interface](#Command-line-interface)
4. [Preprocessing tool](#Preprocessing-tool)
## Introduction

## Command line interface
### Usage

```
python signedAriaDNE.py input [input] [-h] [-v] [-o [OUTPUT]] [-b BANDWIDTH] [-d {Euclidean,Geodesic}] [-c CUTOFF]
```

#### Arguments

- `input`: Path to .ply/.obj file(s) or directory containing mesh files.

If the files are non-watertight, watertight versions may be provided alongside.
The watertight versions are required to end with "_watertight" before the suffix.
For instance, given a file called `tooth.ply`, the watertight version should be named `tooth_watertight.ply`.
Files ending with "_watertight" will NOT be processed individually.

#### Options

- `-h`, `--help`: Show help message and exit
- `-v`, `--visualize`: Visualize 3d mesh colored by normalized local DNE values (only for single file inputs)
- `-o [OUTPUT]`, `--output [OUTPUT]`: Specify output path for results (default: results.csv).
- `-b BANDWIDTH`, `--bandwidth BANDWIDTH`: Set the bandwidth for DNE calculation (default: 0.08)
- `-d {Euclidean,Geodesic}`, `--distance-type {Euclidean,Geodesic}`: Specify the distance type for calculations (default: Euclidean)
- `-c CUTOFF`, `--cutoff CUTOFF`: Set the cut-off threshold for DNE calculation (default: 0)

### Output

The tool outputs the following values for each processed mesh as coloumns:

- File: Path to the input file
- DNE: Overall Dirichlet Normal Energy
- Positive DNE: Positive component of DNE
- Negative DNE: Negative component of DNE

If the `-o` or `--output` flag is off, results will be outputed to STDOUT.

### Visualization

When the `-v` or `--visualize` flag is used with a single input file, the tool will display a 3D visualization of the mesh. The mesh will be colored based on the normalized local DNE values, color gradient from blue through white to red. Blue represents low DNE values and red represents high DNE values.

### Examples

1. Calculate DNE for a single file and visualize:
   ```
   python signedAriaDNE.py path/to/mesh.ply -v
   ```

2. Calculate DNE for multiple files and save results to CSV:
   ```
   python signedAriaDNE.py path/to/mesh1.obj path/to/mesh2.ply -o results.csv
   ```

3. Calculate DNE for all mesh files in a directory with custom bandwidth:
   ```
   python signedAriaDNE.py path/to/mesh/directory -b 0.1
   ```



### Dependencies

- scipy
- numpy
- pandas
- trimesh
- matplotlib

## Preprocessing tool
