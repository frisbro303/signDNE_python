import trimesh
import numpy.matlib
import argparse
from pathlib import Path
import pandas as pd
import sys
from signed_ariaDNE import ariaDNE
import numpy as np


def visualize_mesh(mesh, local_DNE):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    normalized_values = (local_DNE - np.min(local_DNE)) / (np.max(local_DNE) - np.min(local_DNE))
    colors = [(0, 0, 1), (247/255, 240/255, 213/255), (1, 0, 0)]  # Blue, White, Red
    custom_cmap = LinearSegmentedColormap.from_list("custom_bwr", colors)
    colors = custom_cmap(normalized_values)
    mesh.visual.vertex_colors = np.hstack([(colors[:, :3] * 255).astype(np.uint8), np.full((len(mesh.vertices), 1), 1 * 255, dtype=np.uint8)])
    mesh.fix_normals()
    scene = trimesh.Scene(mesh)
    angle = np.radians(120)  # 90 degrees to radians
    rotation_matrix = np.array([
        [1, 0,            0,           0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle),  np.cos(angle), 0],
        [0, 0,            0,           1]
    ])

    # Apply the rotation to the mesh
    mesh.apply_transform(rotation_matrix)
    scene.camera.resolution = [768, 768]
    scene.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Signed Aria DNE 3d meshes.")
    parser.add_argument("input", nargs='+', help="Path to mesh file(s) or directory containing mesh files")
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
    file_names = []
    for path in input_paths:
        p = Path(path)
        if p.is_dir():
            file_names.extend(p.rglob('*'))
        elif p.is_file():
            file_names.append(p)
        else:
            print(str(p) + " is not a file a or a directory")

    # Only return paths that are files and do not have _watertight ending
    return [f for f in file_names if f.is_file() and not has_postfix(f)]


def safe_load(file):
    try:
        # Check if a watertight version exists
        watertight_file = file.with_stem(str(file.stem) + "_watertight")
        if watertight_file.exists():
            return (trimesh.load(str(file)), trimesh.load(str(watertight_file)), file)
        else:
            return (trimesh.load(str(file)), None, file)
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
        print("No files found in the specified input(s).")
        sys.exit(1)

    if args.visualize and len(file_names) != 1:
        print("Visualization only possible for single file inputs. Ignoring flag")
        sys.exit(1)

    # Get sucessfully loaded files and file names
    load = [mesh for mesh in map(safe_load, file_names) if mesh is not None]
    # get meshes from loaded files
    meshes = [(m[0], m[1]) for m in load]
    file_names = [m[2] for m in load]

    values = [ariaDNE(mesh, watertight_mesh, bandwidth=args.bandwidth,
                      cutoff=args.cutoff, distance_type=args.distance_type) \
              for (mesh, watertight_mesh) in meshes]

    df = create_dataframe([v[2:] for v in values], file_names)
    output_results(df, args.output)

    if args.visualize:
        visualize_mesh(meshes[0][0], values[0][1])


if __name__ == '__main__':
    main()
