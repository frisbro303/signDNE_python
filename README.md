# CODE HAS NOT BEEN UPDATED ACCORDING TO THIS NEW DOCUMENTATION

# signedAriaDNE.py

Calculate Signed Aria DNE for PLY and OBJ mesh files.

## Usage

```
signedAriaDNE.py INPUT [OPTIONS]
```

## Arguments

- `INPUT`: Path to a single .ply/.obj file or a directory containing mesh files (required)

## Options

- `-v, --visualize`: Enable visualization (only for single file inputs)
- `-o, --output PATH`: Specify output path for results (CSV file for multiple inputs, or directory for single input)
- `-r, --recursive`: Process subdirectories recursively
- `-b, --bandwidth FLOAT`: Set the bandwidth for DNE calculation (default: 0.1)
- `-d, --distance-type [euclidean|geodesic]`: Specify the distance type for calculations (default: euclidean)
- `-c, --cutoff FLOAT`: Set the cut-off threshold for DNE calculation (default: 2.0)
- `--help`: Show this message and exit

## Examples

```bash
# Process a single file
signedAriaDNE.py mesh.ply

# Process a single file with visualization
signedAriaDNE.py mesh.obj -v

# Process a directory and save results to a CSV file
signedAriaDNE.py mesh_directory -o /path/to/results.csv

# Process a single file with visualization and specify output directory
signedAriaDNE.py mesh.ply -v -o ./output_dir/

# Process a directory recursively and save results to a CSV file
signedAriaDNE.py mesh_directory -r -o results.csv

# Process a single file with custom parameters
signedAriaDNE.py mesh.ply -b 0.2 -d geodesic -c 1.5
```

## Notes

- Visualization (`-v`) is only available for single file inputs
- When processing a directory, only .ply and .obj files will be processed; all other file types are ignored
- Use `-r` for recursive processing of subdirectories
- Files in formats other than .ply or .obj will be automatically ignored during folder processing
- The bandwidth, distance type, and cut-off threshold affect the DNE calculation algorithm
- For multiple inputs, the output (`-o`) should specify a CSV file; for single inputs, it can specify a directory
