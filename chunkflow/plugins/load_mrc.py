from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk

import mrcfile

def execute(bbox: BoundingBox = None, fname: str = None, dtype=None, voxel_size=None, voxel_size_factor=None):

    with mrcfile.mmap(fname) as mrc:
        # print(mrc.header)
        # print(f'volume shape: {mrc.data.shape}')
        # print(f'voxel size: {mrc.voxel_size}')
        if voxel_size is None:
            voxel_size = mrc.voxel_size
            if voxel_size_factor is None:
                voxel_size_factor = 10
        if bbox is None:
            arr = mrc.data
            bbox = BoundingBox.from_delta((0, 0, 0), arr.shape)
        else:
            arr = mrc.data[bbox.slices]

    if isinstance(voxel_size, str):
        try:
            voxel_size = eval(voxel_size)
            if isinstance(voxel_size, (int, float)):
                voxel_size = (voxel_size, voxel_size, voxel_size)
        except SyntaxError:
            voxel_size = tuple(map(float, voxel_size.strip().split(' ')))
            voxel_size_ints = tuple(map(int, voxel_size))
            if voxel_size_ints == voxel_size:
                voxel_size = voxel_size_ints
    voxel_size = Cartesian.from_collection(voxel_size)
    if voxel_size_factor:
        voxel_size = voxel_size / voxel_size_factor
    print(f'voxel size: {voxel_size}')

    if dtype:
        arr = arr.view(dtype)

    chunk = Chunk(arr, voxel_offset=bbox.start, voxel_size=voxel_size)
    return chunk 
