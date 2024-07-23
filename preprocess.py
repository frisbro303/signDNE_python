import pymeshlab
import os
import sys
import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess files for use in signed DNE python package. Fill holes, remove unreferenced vertices, etc...")
    parser.add_argument("input", nargs='+', help="Path to .ply/.obj file(s) or directory containing mesh files")
    return parser.parse_args()


def get_file_names(input_paths):
    file_names = []
    for path in input_paths:
        p = Path(path)
        if p.is_dir():
            file_names.extend(p.glob('*'))
        elif p.is_file():
            file_names.append(p)
        else:
            print(str(p) + " is not a file a or a directory")
    return [f for f in file_names if f.suffix in ('.ply', '.obj')]


def preprocess_file(file_name):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_name)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_folded_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(file_name)
    ms.meshing_close_holes(
            maxholesize=10000000,
            selected=False,
            newfaceselected=True, 
            selfintersection=False,
            refinehole=True,
        )
    base_name, extension = os.path.splitext(file_name)
    new_file_name = f"{base_name}_watertight{extension}"
    ms.save_current_mesh(new_file_name)


def main():
    args = parse_arguments()
    file_names = get_file_names(args.input)

    if not file_names:
        print("No .ply or .obj files found in the specified input(s).")
        sys.exit(1)

    for file_name in file_names:
        print(file_name)
        preprocess_file(str(file_name))

    print("filled holes on " + str(len(file_names)) + " files")


if __name__ == '__main__':
    main()
