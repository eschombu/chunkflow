from ast import literal_eval
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional

import numpy as np

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk


def simplest_type(s: str):
    try:
        return literal_eval(s)
    except:
        return s


def str_to_dict(string: str):
    keywords = {}
    for item in string.split(';'):
        assert '=' in item
        item = item.split('=')
        keywords[item[0]] = simplest_type(item[1])
    return keywords


def infer_bbox(
        task: dict,
        chunk_start: Optional[Cartesian] = None,
        chunk_size: Optional[Cartesian] = None,
) -> BoundingBox:
    chunk_bboxes = [c.bbox for c in task.values() if isinstance(c, Chunk)]
    bbox_min_max = (min([bb.minpt for bb in chunk_bboxes]), max([bb.maxpt for bb in chunk_bboxes]))
    if chunk_start is None:
        chunk_start = bbox_min_max[0]
    if chunk_size is None:
        chunk_size = bbox_min_max[1] - bbox_min_max[0]
    return BoundingBox.from_delta(chunk_start, chunk_size)


@dataclass
class SharedMemoryContainer:
    shared_mem: shared_memory.SharedMemory
    shape: tuple
    dtype: type

    @classmethod
    def create(cls, array: np.ndarray):
        shared_mem = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shared_mem.buf)
        np.copyto(shared_array, array)
        return cls(shared_mem, array.shape, array.dtype)

    def load(self):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shared_mem.buf)

    def close(self):
        self.shared_mem.close()

    def unlink(self):
        self.shared_mem.unlink()
