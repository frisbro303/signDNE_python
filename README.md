# Signed ariaDNE CLI
A python package for robustly computing DNE and signedDNE

## Index
1. [Introduction](#introduction)
2. [Install](#install)
3. [Command line interface](#Command-line-interface)
4. [Preprocessing tool](#Preprocessing-tool)
## Introduction
The package consists of 3 files
- `src/signed_ariaDNE.py` : File containing the function `ariaDNE` for calculating the DNE and signed DNE of a shape, which can be imported as a library or used through the command line interface. 
- `src/signed_ariaDNE_cli.py` : Command line interface for the `ariaDNE` function.
- `src/preprocess.py` : Script for doing simple cleanups and generating watertight version of meshes.

## Command line interface
-- description go here

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

- File name
- DNE
- PPositive component of DNE
- Negative component of DNE

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
-- description goes here.
- removing duplicate faces and vertices
- 
### Usage

```
python signedAriaDNE.py input [input] [-h] [-w]
```

#### Arguments

- `input`: Path to .ply/.obj file(s) or directory containing mesh files.


#### Options

- `-h`, `--help`: Show help message and exit
- `-w`, `--watertight` : Generate a watertight version for each mesh ending in "_watertight" before the final suffix.

### Examples

1. Preprocess one file and generate a watertight version:
   ```
   python preprocess.py path/to/non-watertight-mesh.ply -w
   ```

2. Preprocessing multiple files.
   ```
   python signedAriaDNE.py path/to/mesh1.obj path/to/mesh2.ply
   ```

3. Preprocessing files in a folder and generate watertigt versions:
   ```
   python signedAriaDNE.py path/to/mesh/directory --watertight
   ```

### Dependencies

- pymeshlab
