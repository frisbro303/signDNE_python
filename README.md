## Usage

Run the script from the command line with the following syntax:

```
python signedAriaDNE.py <path> [-v] [-export=<filename>]
```

Arguments:
- `<path>`: Path to a single .ply file or a directory containing .ply files
- `-v`: (Optional) Enable visualization (only works for single file inputs)
- `-export=<filename>`: (Optional) Export results to a CSV file

Examples:
```
# Process a single file
python signedAriaDNE.py /path/to/mesh.ply

# Process a single file with visualization
python signedAriaDNE.py /path/to/mesh.ply -v

# Process a directory and export results
python signedAriaDNE.py /path/to/mesh/directory -export=results.csv

# Process a single file with visualization and export
python signedAriaDNE.py /path/to/mesh.ply -v -export=results.csv
```

## Output

The script outputs the following:
- Console messages indicating processing status
- (Optional) CSV file with columns: DNE, Positive DNE, Negative DNE
- (Optional) Mesh visualization for single file inputs

## Notes

- Visualization is only available for single file inputs
- When processing a directory, only .ply files will be considered
- Ensure you have the necessary permissions to read input files and write to the export location
