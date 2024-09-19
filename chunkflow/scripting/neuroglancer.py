import argparse
import os
import subprocess

FILE_VAR_SEP = '#'  # Separator between file path and variable name in a single string.
PRECOMPUTED_PREFIX = 'precomputed://'
FILE_PREFIX = 'file://'


def _is_hdf5(file_path: str):
    return file_path.lower().endswith('.h5') or file_path.lower().endswith('.hdf5')


def _is_precomputed(file_path: str):
    return file_path.lower().startswith('precomputed://')


def _is_tiff(file_path: str):
    return file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif')


def _is_png(file_path: str):
    return file_path.lower().endswith('.png')


def _load_hdf5(file_path: str, var_name: str, infer_chunk: bool = True):
    cmds = f'load-h5 -f "{file_path}"'
    if var_name:
        cmds += f' -o {var_name}'
    return cmds


def _load_precomputed(file_path: str, var_name: str, infer_chunk: bool = True):
    if file_path.startswith(PRECOMPUTED_PREFIX):
        file_path = file_path[len(PRECOMPUTED_PREFIX):]
    if not file_path.startswith(FILE_PREFIX):
        file_path = FILE_PREFIX + file_path
    cmds = f'load-precomputed -v "{file_path}"'
    if infer_chunk:
        cmds += ' --infer-chunk'
    if var_name:
        cmds += f' -o {var_name}'
    return cmds


def _load_tiff(file_path: str, var_name: str, infer_chunk: bool = True):
    cmds = f'load-tif -f "{file_path}"'
    if infer_chunk:
        cmds += ' --infer-chunk'
    if var_name:
        cmds += f' -o {var_name}'
    return cmds


def _load_png(file_path: str, var_name: str, infer_chunk: bool = True):
    cmds = f'load-png -f "{file_path}"'
    if infer_chunk:
        cmds += ' --infer-chunk'
    if var_name:
        cmds += f' -o {var_name}'
    return cmds


def _parse_file_vars(file_path: str):
    if FILE_VAR_SEP in file_path:
        file_path, var_name = file_path.split(FILE_VAR_SEP)
    else:
        var_name = os.path.splitext(os.path.basename(file_path))[0]
    if ',' in var_name:
        raise ValueError('Variable name cannot contain a comma.')
    return file_path, var_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', type=str, nargs='*', help='Path(s) to data file(s).')
    parser.add_argument('--precomputed', type=str, nargs='*', help='Path(s) to precomputed volume(s).')
    parser.add_argument('--hdf5', type=str, nargs='*', help='Path(s) to HDF5 file(s).')
    parser.add_argument('--h5', type=str, nargs='*', help='Path(s) to HDF5 file(s). (Alias for --hdf5)')
    parser.add_argument('--tiff', type=str, nargs='*', help='Path(s) to TIFF file(s).')
    parser.add_argument('--png', type=str, nargs='*', help='Path(s) to PNG file(s).')
    parser.add_argument('--infer-off', action='store_true', help='Disable automatic chunk inference.')

    bbox_coord_args = ['start', 'stop', 'center', 'size']
    for arg in bbox_coord_args:
        parser.add_argument(f'--{arg}', type=int, nargs=3, default=None, help=f'Chunk {arg} coordinates.')
    parser.add_argument('--xyz', action='store_true')
    parser.add_argument('--bbox', type=str, default=None, help=f'BoundingBox string.')

    args = parser.parse_args()

    cmds = ['chunkflow']

    bbox_arg_vals = []
    for arg in bbox_coord_args:
        arg_coords = getattr(args, arg)
        if arg_coords:
            if args.xyz:
                arg_coords = arg_coords[::-1]
            coords_str = ' '.join(str(x) for x in arg_coords)
            bbox_arg_vals.append(f'--{arg} {coords_str}')
    if args.bbox:
        bbox_arg_vals.append(f'--string {args.bbox}')
    if bbox_arg_vals:
        cmds.append('create-bbox ' + ' '.join(bbox_arg_vals))

    infer_chunk = not args.infer_off

    vars = []
    for file in args.files or []:
        file, var_name = _parse_file_vars(file)
        if _is_precomputed(file):
            load_fun = _load_precomputed
        elif _is_hdf5(file):
            load_fun = _load_hdf5
        elif _is_tiff(file):
            load_fun = _load_tiff
        elif _is_png(file):
            load_fun = _load_png
        else:
            raise ValueError(f'Unsupported file type: {file}')
        vars.append(var_name)
        cmds.append(load_fun(file, var_name, infer_chunk=infer_chunk))
    for file in args.precomputed or []:
        path, var_name = _parse_file_vars(file)
        vars.append(var_name)
        cmds.append(_load_precomputed(path, var_name, infer_chunk=infer_chunk))
    for file in ((args.hdf5 or []) + (args.h5 or [])):
        path, var_name = _parse_file_vars(file)
        vars.append(var_name)
        cmds.append(_load_hdf5(path, var_name, infer_chunk=infer_chunk))
    for file in args.tiff or []:
        path, var_name = _parse_file_vars(file)
        vars.append(var_name)
        cmds.append(_load_tiff(path, var_name, infer_chunk=infer_chunk))
    for file in args.png or []:
        path, var_name = _parse_file_vars(file)
        vars.append(var_name)
        cmds.append(_load_png(path, var_name, infer_chunk=infer_chunk))

    cmds.append(f"neuroglancer -i {','.join(vars)}")
    cmd = ' \\\n    '.join(cmds)
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    print()


if __name__ == '__main__':
    main()
