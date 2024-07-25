from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh
import numpy as np
import numpy.matlib
import argparse
from pathlib import Path
import pandas as pd
import sys
from signed_ariaDNE import ariaDNE


def visualize_mesh(mesh, local_DNE):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    normalized_values = (local_DNE - np.min(local_DNE)) / (np.max(local_DNE) - np.min(local_DNE))
    colors = [(0, 0, 1), (247/255, 240/255, 213/255), (1, 0, 0)]  # Blue, White, Red
    custom_cmap = LinearSegmentedColormap.from_list("custom_bwr", colors)
    colors = custom_cmap(normalized_values)
    mesh.visual.vertex_colors = np.hstack([(colors[:, :3] * 255).astype(np.uint8), np.full((len(mesh.vertices), 1), 1 * 255, dtype=np.uint8)])
    mesh.fix_normals()
    mesh.show() 


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Signed Aria DNE for PLY and OBJ mesh files.")
    parser.add_argument("input", nargs='+', help="Path to .ply/.obj file(s) or directory containing mesh files")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization (only for single file inputs)")
    parser.add_argument("-o", "--output", nargs='?', const="results.csv", default=None, help="Specify output path for results")
    parser.add_argument("-b", "--bandwidth", type=float, default=0.08, help="Set the bandwidth for DNE calculation (default: 0.08)")
    parser.add_argument("-d", "--distance-type", choices=['Euclidean', 'Geodesic'], default='Euclidean', help="Specify the distance type for calculations (default: Euclidean)")
    parser.add_argument("-c", "--cutoff", type=float, default=0, help="Set the cut-off threshold for DNE calculation (default: 0)")
    return parser.parse_args()


def has_postfix(file):
    parts = file.stem.split('_')
    return parts[-1] == 'watertight'


def get_file_names(input_paths):
    # Find paths of all given files and files in folders
    initial_paths = []
    for path in input_paths:
        p = Path(path)
        if p.is_dir():
            initial_paths.extend(p.glob('*'))
        elif p.is_file():
            initial_paths.append(p)
        else:
            print(str(p) + " is not a file a or a directory")

    files = [Path(f) for f in initial_paths if Path(f).suffix in ('.ply', '.obj')]

    # Only load files that do not have _watertight postfix
    files = [f for f in files if not has_postfix(f)]

    return files


def safe_load(file):
    try:
        # Check if a watertight version exists
        watertight_file = file.with_stem(str(file.stem) + "_watertight")
        if watertight_file.exists():
            return (trimesh.load(str(file)), trimesh.load(str(watertight_file)))
        else:
            return (trimesh.load(str(file)), None)
    except Exception as e:
        print(e)
        return None


def create_dataframe(data, file_names):
    df = pd.DataFrame(data, columns=["DNE", "positive DNE", "negative DNE"])
    df["File"] = [str(f) for f in file_names]
    return df[["File", "DNE", "positive DNE", "negative DNE"]]


def output_results(df, output_file):
    if output_file:
        df.to_csv(output_file, index=False, float_format='%.16f')
        print(f"Results saved to {output_file}")
    else:
        print(df.to_string(index=False, float_format=lambda x: f'{x:.16f}'))


def main():
    args = parse_arguments()
    file_names = get_file_names(args.input)

    if not file_names:
        print("No .ply or .obj files found in the specified input(s).")
        sys.exit(1)

    if args.visualize and len(file_names) != 1:
        print("Visualization only possible for single file inputs. Ignoring flag")
        sys.exit(1)

    meshes = [mesh for mesh in map(safe_load, file_names) if mesh is not None]

    values = [ariaDNE(mesh, watertight_mesh, bandwidth=args.bandwidth, 
                      cutoff=args.cutoff, distance_type=args.distance_type) \
              for (mesh, watertight_mesh) in meshes]

    df = create_dataframe([v[1:] for v in values], file_names)
    output_results(df, args.output)

    if args.visualize:
        visualize_mesh(meshes[0][0], values[0][0])


if __name__ == '__main__':
    main()
