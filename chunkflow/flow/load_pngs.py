import os

import numpy as np
import pyspng
from itertools import repeat
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk


def load_png_image(file_name: str):
    with open(file_name, "rb") as f:
        arr = pyspng.load(f.read())
    if np.ndim(arr) == 3:
        arr = arr[:, :, 0]
    return arr

def _par_load(file_name, bbox, dtype):
    arr = load_png_image(file_name)
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr[bbox.start[1]:bbox.stop[1], bbox.start[2]:bbox.stop[2]]


def load_png_images(
        path_prefix: str, 
        bbox: BoundingBox = None, 
        voxel_offset: Cartesian = Cartesian(0, 0, 0),
        voxel_size: Cartesian = Cartesian(1, 1, 1),
        digit_num: int = 5,
        dtype: np.dtype = None,
        layer_type: str = 'image',
        workers: int = 1,
        tqdm_chunksize: int = None,
):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    path_prefix = os.path.expanduser(path_prefix)
    if os.path.isfile(path_prefix):
        dir_path = os.path.dirname(path_prefix)
        all_png_filenames = [os.path.basename(path_prefix)]
    else:
        if os.path.isdir(path_prefix):
            if not path_prefix.endswith('/'):
                path_prefix += '/'
            dir_path = path_prefix
        else:
            dir_path = os.path.dirname(path_prefix)
        all_png_filenames = sorted(fname for fname in os.listdir(dir_path) if fname.endswith('.png'))

    if bbox is None:
        file_names = [os.path.join(dir_path, fname) for fname in all_png_filenames]
        arr = load_png_image(file_names[0])
        shape = Cartesian(len(file_names), arr.shape[0], arr.shape[1])
        bbox = BoundingBox.from_delta(voxel_offset, shape)
    elif len(all_png_filenames) == bbox.shape[0]:
        file_names = [os.path.join(dir_path, fname) for fname in all_png_filenames]
    else:
        # Allow for a path prefix and a bbox to determine which png files to load
        file_names = []
        for z in tqdm(range(bbox.start[0], bbox.stop[0])):
            file_name = f'{path_prefix}{z:0>{digit_num}d}.png'
            if os.path.exists(file_name):
                file_names.append(file_name)
            else:
                print(f'Warning: {file_name} does not exist')

    chunk = Chunk.from_bbox(
        bbox,
        dtype=dtype,
        pattern='zero', 
        voxel_size=voxel_size,
    )

    # Use available CPU cores minus if workers is negative (minus n if workers == -(n+1))
    if workers is not None and workers < 0:
        workers = os.cpu_count() + 1 + workers

    tqdm_kws = dict(desc=f'Loading PNGs', total=len(file_names))
    if workers and workers > 1:
        if tqdm_chunksize is None and len(file_names) >= 1024:
            tqdm_kws['chunksize'] = 32
        arrs = process_map(_par_load, file_names, repeat(bbox), repeat(dtype), max_workers=workers, **tqdm_kws)
        chunk.array = np.stack(arrs, axis=0)
    else:
        for z_offset, file_name in tqdm(enumerate(file_names), **tqdm_kws):
            arr = load_png_image(file_name)
            if arr.dtype != dtype:
                arr = arr.astype(dtype)

            chunk.array[z_offset, :, :] = arr[
                bbox.start[1]:bbox.stop[1],
                bbox.start[2]:bbox.stop[2]]

    chunk.layer_type = layer_type if layer_type is not None else 'unknown'
    return chunk
