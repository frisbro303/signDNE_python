# Signed Aria DNE Calculator

This tool calculates the Signed DNE for 3d meshes. It provides options for visualization and customization of the calculation parameters.

## Usage

```
python signedAriaDNE.py input [input] [-h] [-v] [-o [OUTPUT]] [-b BANDWIDTH] [-d {Euclidean,Geodesic}] [-c CUTOFF]
```

### Arguments

- `input`: Path to .ply/.obj file(s) or directory containing mesh files

### Options

- `-h`, `--help`: Show help message and exit
- `-v`, `--visualize`: Visualize 3d mesh colored by normalized local DNE values (only for single file inputs)
- `-o [OUTPUT]`, `--output [OUTPUT]`: Specify output path for results (default: results.csv).
- `-b BANDWIDTH`, `--bandwidth BANDWIDTH`: Set the bandwidth for DNE calculation (default: 0.08)
- `-d {Euclidean,Geodesic}`, `--distance-type {Euclidean,Geodesic}`: Specify the distance type for calculations (default: Euclidean)
- `-c CUTOFF`, `--cutoff CUTOFF`: Set the cut-off threshold for DNE calculation (default: 0)

## Output

The tool outputs the following values for each processed mesh as coloumns:

- File: Path to the input file
- DNE: Overall Dirichlet Normal Energy
- Positive DNE: Positive component of DNE
- Negative DNE: Negative component of DNE

If the `-o` or `--output` flag is off, results will be outputed to STDOUT.

## Examples

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


## Dependencies

- scipy
- open3d
- numpy
- pandas
- trimesh
- matplotlib
