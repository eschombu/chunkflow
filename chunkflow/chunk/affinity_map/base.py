__doc__ = """Image chunk class"""

import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.chunk import Chunk


class AffinityMap(Chunk):
    """
    a chunk of affinity map. It has x,y,z three channels with single precision.
    """
    def __init__(self, array: np.ndarray,
            voxel_offset: Cartesian=None, 
            voxel_size: Cartesian=None,
            layer_type: str = None):
        assert isinstance(array, np.ndarray)
        assert array.ndim == 4
        assert np.issubdtype(array.dtype, np.float32)
        assert array.shape[0] == 3
        super().__init__(array, 
            voxel_offset=voxel_offset, 
            voxel_size=voxel_size, 
            layer_type=layer_type)

    @classmethod
    def from_chunk(cls, chk: Chunk):
        assert isinstance(chk, Chunk)
        return cls(chk.array, 
            voxel_offset=chk.voxel_offset,
            voxel_size=chk.voxel_size,
            layer_type=chk.layer_type)

    def quantize(self, mode: str = 'xy'):
        """transform affinity map to gray scale image

        Args:
            mode (str, optional): tranformation mode. Defaults to 'xy'.

        Raises:
            ValueError: only support mode of xy and z.

        Returns:
            Chunk: the gray scale image chunk
        """
        if mode == 'mean' or set(mode) == {'x', 'y', 'z'}:
            image = self.array.mean(axis=0)
        elif mode == 'max':
            image = self.array.max(axis=0)
        elif mode == 'min':
            image = self.array.min(axis=0)
        elif (0 < len(mode) < 3) and all(ax in {'x', 'y', 'z'} for ax in mode):
            ix_map = {'x': 0, 'y': 1, 'z': 2}
            image = np.zeros(self.shape[1:], dtype=np.float32)
            for ax in mode:
                image += self[ix_map[ax], ...]
            image /= len(mode)
        else:
            raise ValueError(f"Invalid value for mode: '{mode}'")

        image = (image * 255.).astype(np.uint8)
        image = Chunk(image)
        image.set_properties(self.properties)
        image.layer_type = None
        # assert np.issubdtype(image.dtype, np.uint8)
        return image