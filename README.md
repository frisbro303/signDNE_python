# Sign ariaDNE
A Python package for ariaDNE and its sign-oriented extension.

## Index
1. [Package description](#Package-description)
2. [Installation](#Installation)
3. [Signed ariaDNE](#Signed-ariaDNE)
4. [Command line interface](#Command-line-interface)

## Installation
In order to install the library and command line interface run the following command:
```
pip install git+https://github.com/anonymized/for/review.git
```

## sign_ariaDNE library
The library can be imported as `from sign_ariaDNE import ariaDNE`.

The function will calculate local DNE, local curvature values, DNE, positiveDNE component, and negativeDNE component.

The following preprocessing is automatically performed:
- Closing of all holes if given mesh is non-watertight.
- remove degenerate faces
- remove duplicate faces
- remove unreferenced vertices
- remove infinite values from face and vertex data
  

## Command line interface
Command line interface for the `ariaDNE` function.

### Usage
```
sign_ariaDNE input [input] [-h] [-v] [-o [OUTPUT]] [-b BANDWIDTH] [-d {Euclidean,Geodesic}] [-c CUTOFF]
```

#### Arguments

- `input`: Path to mesh file(s) or directory containing mesh files. Files should be PLY or OBJ format. 

#### Options

- `-h`, `--help`: Show help message and exit
- `-v`, `--visualize`: Visualize 3d mesh colored by normalized local DNE values (only for single file inputs)
- `-o [OUTPUT]`, `--output [OUTPUT]`: Specify output path for results (default: results.csv).
- `-b BANDWIDTH`, `--bandwidth BANDWIDTH`: Set the bandwidth for DNE calculation (default: 0.08)
- `-d {Euclidean,Geodesic}`, `--distance-type {Euclidean,Geodesic}`: Specify the distance type for calculations (default: Euclidean)
- `-c CUTOFF`, `--cutoff CUTOFF`: Set the cut-off threshold for DNE calculation (default: 0)

### Output

The CLI outputs the following values for each processed mesh as coloumns:

- File name
- DNE
- Positive component of DNE
- Negative component of DNE

If the `-o` or `--output` flag is off, results will be outputed to STDOUT.

### Visualization

When the `-v` or `--visualize` flag is used with a single input file, the tool will display a 3D visualization of the mesh. The mesh will be colored based on the normalized local DNE values, color gradient from blue through white to red. Blue represents low DNE values and red represents high DNE values.

### Examples

1. Calculate signed ariaDNE for a single file and visualize:
   ```
   sign_ariaDNE path/to/mesh.ply -v
   ```

2. Calculate signed ariaDNE for multiple files and save results to CSV:
   ```
   sign_ariaDNE path/to/mesh1.obj path/to/mesh2.ply -o results.csv
   ```

3. Calculate signed ariaDNE for all mesh files in a directory with custom bandwidth:
   ```
   sign_ariaDNE path/to/mesh/directory -b 0.1
   ```



## Dependencies
- scipy
- trimesh
- numpy
- pyvista
- pandas
- networkx
- rtree
- pyglet<2

