import os


import numpy as np

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk

from tqdm import tqdm
import pyspng


def load_image(file_name: str):
    with open(file_name, "rb") as f:
        img = pyspng.load(f.read())
    # img = np.expand_dims(img, axis=0)
    return img


def load_png_images(
        path_prefix: str, 
        bbox: BoundingBox = None, 
        voxel_offset: Cartesian = Cartesian(0, 0, 0),
        voxel_size: Cartesian = Cartesian(1, 1, 1),
        digit_num: int = 5,
        dtype: np.dtype = np.uint8):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    if os.path.isdir(path_prefix):
        if not path_prefix.endswith('/'):
            path_prefix += '/'
        dir_path = path_prefix
    else:
        dir_path = os.path.dirname(path_prefix)
    all_png_filenames = [fname for fname in os.listdir(dir_path) if fname.endswith('.png')]

    if bbox is None:
        file_names = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith('.png'):
                fname = os.path.join(dir_path, fname)
                file_names.append(os.path.expanduser(fname))

        img = load_image(file_names[0])
        shape = Cartesian(len(file_names), img.shape[0], img.shape[1])
        bbox = BoundingBox.from_delta(voxel_offset, shape)
    elif len(all_png_filenames) == bbox.shape[0]:
        file_names = [os.path.expanduser(os.path.join(dir_path, fname)) for fname in sorted(all_png_filenames)]
    else:
        file_names = []
        for z in tqdm(range(bbox.start[0], bbox.stop[0])):
            file_name = f'{path_prefix}{z:0>{digit_num}d}.png'
            file_name = os.path.expanduser(file_name)
            file_names.append(file_name)

    chunk = Chunk.from_bbox(
        bbox, dtype=dtype, 
        pattern='zero', 
        voxel_size=voxel_size
    )

    for z_offset, file_name in tqdm(enumerate(file_names)):
        if os.path.exists(file_name):
            img = load_image(file_name)
            img = img.astype(dtype=dtype)
            chunk.array[z_offset, :, :] = img[bbox.start[1]:bbox.stop[1], bbox.start[2]:bbox.stop[2]]
        else:
            print(f'image file do not exist: {file_name}')

    return chunk
