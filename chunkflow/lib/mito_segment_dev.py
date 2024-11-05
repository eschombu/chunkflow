import json
import multiprocessing
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from glob import glob
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Self, Union
from warnings import warn

import cc3d
import fastremap
import h5py
import numpy as np
from tqdm import tqdm

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian, Direction, get_connectivity_directions, \
    to_cartesian
from chunkflow.lib.utils import SharedMemoryContainer

Number = Union[int, float]


@dataclass
class SegmentationParams:
    threshold: Optional[Number] = None
    min_size: Optional[int] = None
    min_value: Optional[Number] = None


BOUNDARY_THRESHOLD = 2
DEFAULT_MITO_SEG_PARAMS = SegmentationParams(threshold=0.05, min_size=2000, min_value=0.5)
DEFAULT_GRANULE_SEG_PARAMS = SegmentationParams(threshold=0.05, min_size=200, min_value=0.5)
TMP_DIR = Path('/mnt/ceph/neuro/wasp_em/eschomburg/tmp/sample3/mito_granule')


def _as_proba(array: Union[np.ndarray, Chunk]) -> Union[np.ndarray, Chunk]:
    if isinstance(array, Chunk):
        chunk = array.copy()
        array = chunk.array
    else:
        chunk = None
    if array.dtype == np.uint8:
        array = array.astype(float) / 255
    elif array.max() > 1:
        array = array.astype(float) / array.max()
    if chunk is None:
        return array.copy()
    else:
        chunk.array = array.copy()
        return chunk


def _apply_threshold(array: np.ndarray, threshold: Number, op: Optional[str] = None) -> np.ndarray:
    """
    Make sure array values and threshold values on same scale (e.g., array may be uint8 from 0 to 255 and threshold a
    float probability from 0.0 to 1.0)
    """
    if (0 < threshold < 1) and (array.max() > 1):
        array = _as_proba(array)
    if op is None:
        if threshold == array.min():
            op = '>'
        else:
            op = '>='
    array = eval(f'array {op} {threshold}')
    return array


def threshold_segmentation(
        array: np.ndarray,
        threshold: Union[int, float] = 0,
        op: Optional[str] = None,
        connectivity: int = 6,
) -> np.ndarray:
    """
    Segment the input array using a threshold and comparison operator (default: '>=' for threshold greater than minimum
    array value, '>' for threshold equal to minimum) to separate foreground from background. The connectivity is the
    number of neighbors to consider when finding the resulting connected components.
    """
    array = _apply_threshold(array, threshold, op)
    return cc3d.connected_components(array, connectivity=connectivity)


@dataclass
class AltOverlapInfo:
    overlap_count: int
    overlap_frac: float
    overlap_max: Number
    seg_params: Optional[SegmentationParams] = None

    _overlap_params = ('overlap_count', 'overlap_frac', 'overlap_max')

    def to_h5_dict(self) -> dict:
        d = {param: getattr(self, param) for param in self._overlap_params}
        for k, v in asdict(self.seg_params).items():
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_h5_dict(cls, d: dict) -> Self:
        ovlp_params = {k: v for k, v in d.items() if k in cls._overlap_params}
        seg_params = SegmentationParams(**{k: v for k, v in d.items() if k not in cls._overlap_params})
        return cls(**ovlp_params, seg_params=seg_params)

    def copy(self) -> Self:
        return AltOverlapInfo(
            self.overlap_count,
            self.overlap_frac,
            self.overlap_max,
            SegmentationParams(**asdict(self.seg_params)) if self.seg_params else None,
        )


@dataclass
class SegmentMask:
    id: int
    size: int
    bbox: BoundingBox
    bbox_in_chunk: BoundingBox
    chunk_bbox: BoundingBox
    margin: int = 0
    mask: Optional[np.ndarray[np.bool_]] = None
    neu_seg_overlaps: Optional[Dict[int, float]] = None
    alt_overlap_info: Optional[AltOverlapInfo] = None

    def __repr__(self):
        params = {attr.name: getattr(self, attr.name) for attr in fields(self) if attr.name not in ['mask']}
        params['mask'] = f"[{'x'.join(map(str, self.mask.shape))}]"
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in params.items())})"

    def copy(self, drop_mask=False) -> 'SegmentMask':
        return SegmentMask(
            self.id,
            self.size,
            self.bbox.copy(),
            self.bbox_in_chunk.copy(),
            self.chunk_bbox.copy(),
            self.margin,
            self.mask.copy() if self.mask is not None and not drop_mask else None,
            self.neu_seg_overlaps.copy() if self.neu_seg_overlaps else None,
            self.alt_overlap_info.copy() if self.alt_overlap_info else None,
        )

    @classmethod
    def from_chunk_mask(
            cls,
            seg_id: int,
            mask: np.ndarray[np.bool_],
            chunk_bbox: Optional[BoundingBox] = None,
            margin: int = 0,
            store_mask=True,
    ):
        if chunk_bbox is None:
            chunk_bbox = BoundingBox(Cartesian(0, 0, 0), to_cartesian(mask.shape))
        else:
            assert chunk_bbox.shape == mask.shape
        bbox_in_chunk = BoundingBox.from_points(np.argwhere(mask))
        bbox = bbox_in_chunk + chunk_bbox.start
        if store_mask:
            mask = mask[*bbox_in_chunk.slices].astype(np.bool_)
        else:
            mask = None
        return cls(seg_id, mask.sum(), bbox, bbox_in_chunk, chunk_bbox, margin, mask)

    def save(self, path: Union[str, Path]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('mask', data=self.mask)
            f.attrs['id'] = self.id
            f.attrs['size'] = self.size
            f.attrs['bbox'] = (self.bbox.start, self.bbox.stop)
            f.attrs['bbox_in_chunk'] = (self.bbox_in_chunk.start, self.bbox_in_chunk.stop)
            f.attrs['chunk_bbox'] = (self.chunk_bbox.start, self.chunk_bbox.stop)
            f.attrs['margin'] = self.margin
            if self.neu_seg_overlaps is not None:
                f.attrs['neu_seg_overlaps'] = tuple(self.neu_seg_overlaps.items())
            if self.alt_overlap_info is not None:
                for k, v in self.alt_overlap_info.to_h5_dict().items():
                    if v is not None:
                        key = f'overlap_info/{k}'
                        f.attrs[key] = v

    @staticmethod
    def load_attr(path: Union[str, Path], key: str, check_ext=True):
        if check_ext and os.path.splitext(path)[-1] not in ('.h5', '.hdf5'):
            raise ValueError(f'The file {path} is not an HDF5 file.')
        with h5py.File(path, 'r') as f:
            return f.attrs.get(key, None)

    @classmethod
    def load(cls, path: Union[str, Path], load_mask=True, check_ext=True):
        if check_ext and os.path.splitext(path)[-1] not in ('.h5', '.hdf5'):
            raise ValueError(f'The file {path} is not an HDF5 file.')
        with h5py.File(path, 'r') as f:
            dset = f['mask']
            if load_mask:
                mask = dset[()]
            else:
                mask = None
            seg_id = f.attrs['id']
            size = f.attrs['size']
            bbox = BoundingBox(*[Cartesian.from_collection(a) for a in f.attrs['bbox']])
            bbox_in_chunk = BoundingBox(*[Cartesian.from_collection(a) for a in f.attrs['bbox_in_chunk']])
            chunk_bbox = BoundingBox(*[Cartesian.from_collection(a) for a in f.attrs['chunk_bbox']])
            margin = f.attrs['margin']
            neu_seg_overlaps = None
            if 'neu_seg_overlaps' in f.attrs:
                neu_seg_overlaps = dict(f.attrs['neu_seg_overlaps'])
            overlap_attrs = [k for k in f.attrs.keys() if k.startswith('overlap_info/')]
            overlap_info = {}
            if overlap_attrs:
                for attr in overlap_attrs:
                    k = attr.split('/', maxsplit=1)[1]
                    overlap_info[k] = f.attrs[attr]
            overlap_info = AltOverlapInfo.from_h5_dict(overlap_info) if overlap_info else None
        return cls(seg_id, size, bbox, bbox_in_chunk, chunk_bbox, margin, mask, neu_seg_overlaps, overlap_info)

    def on_chunk_boundaries(self, margin: Optional[int] = None, threshold: int = BOUNDARY_THRESHOLD) -> List[Direction]:
        """
        Check if the segment mask is on the boundaries of the chunk. Return the directions of the boundaries that the
        segment mask touches.
        """
        if self.mask is None:
            raise ValueError('The mask has not been loaded.')
        if margin is None:
            if self.margin > 0:
                margin = self.margin
            else:
                margin = 1
        boundary_dirs = []
        for direction in get_connectivity_directions(6):
            assert direction.magnitude == 1
            axis = np.argmax(np.abs(direction))
            d = direction[axis]
            margin_bbox = None
            margin_start = [0, 0, 0]
            margin_stop = self.chunk_bbox.shape.copy().list
            if d < 0 and self.bbox_in_chunk.start[axis] < margin:
                # [0 --> margin] along axis direction, [0 --> stop] along other directions
                margin_stop[axis] = margin
                margin_bbox = BoundingBox(to_cartesian(margin_start), to_cartesian(margin_stop))
            elif d > 0 and self.bbox_in_chunk.stop[axis] >= (self.chunk_bbox.shape[axis] - margin):
                # [(stop - margin) --> stop] along axis direction, [0 --> stop] along other directions
                margin_start[axis] = self.chunk_bbox.shape[axis] - margin
                margin_bbox = BoundingBox(to_cartesian(margin_start), to_cartesian(margin_stop))
            if margin_bbox is not None:
                mask_margin_bbox = self.bbox_in_chunk.intersection(margin_bbox)
                if self.chunk_mask[*mask_margin_bbox.slices].sum() >= threshold:
                    boundary_dirs.append(direction)
        return boundary_dirs

    @property
    def chunk_mask(self):
        """
        Expanded mask with the shape of the chunk.
        """
        if self.mask is None:
            raise ValueError('The mask has not been loaded.')
        mask = np.zeros(self.chunk_bbox.shape, dtype=np.bool_)
        mask[*self.bbox_in_chunk.slices] = self.mask
        return mask

    def crop(self, crop_margin: Union[int, Cartesian, BoundingBox]) -> Optional['SegmentMask']:
        """
        Crop the mask to a smaller bounding box. If the cropped bbox is empty, return None.
        """
        if not isinstance(crop_margin, BoundingBox):
            if not hasattr(crop_margin, '__len__'):
                crop_margin = [crop_margin] * 3
            if not isinstance(crop_margin, Cartesian):
                crop_margin = to_cartesian(crop_margin)
            crop_margin = BoundingBox(crop_margin, self.chunk_bbox.shape - crop_margin)

        # Make sure that the crop_margin is within the chunk_bbox; if it is empty, return None
        crop_margin = crop_margin.intersection(BoundingBox(Cartesian(0, 0, 0), self.chunk_bbox.shape))
        if crop_margin.size == 0:
            return None

        # Figure out if we can set a new `margin` int
        if (
                not all(m == crop_margin.start[0] for m in crop_margin.start)
                or not all((s - m) == crop_margin.start[0] for m, s in zip(crop_margin.stop, self.chunk_bbox.shape))
        ):
            # if self.margin:
            #     warn('crop margin is not uniform, returned cropped object will not preserve margin attribute')
            new_margin = 0
        else:
            new_margin = max(0, self.margin - crop_margin.start[0])

        new_chunk_bbox = crop_margin + self.chunk_bbox.start

        # If the cropped bbox is the same as the original, return a copy of this object
        if new_chunk_bbox == self.chunk_bbox:
            return self.copy()

        # If the segment is outside the cropped bbox, return None
        if new_chunk_bbox.size == 0:
            return None
        new_bbox = self.bbox.intersection(new_chunk_bbox)
        if new_bbox.size == 0:
            return None

        new_bbox_in_orig_chunk = new_bbox - self.chunk_bbox.start
        new_bbox_in_chunk = new_bbox - new_chunk_bbox.start
        new_mask = self.chunk_mask[*new_bbox_in_orig_chunk.slices]
        return SegmentMask(
            id=self.id,
            size=new_mask.sum(),
            bbox=new_bbox,
            bbox_in_chunk=new_bbox_in_chunk,
            chunk_bbox=new_chunk_bbox,
            margin=new_margin,
            mask=new_mask,
        )

    def intersection(self, bbox: BoundingBox) -> Optional['SegmentMask']:
        """
        Get the intersection of the mask with the given bounding box. If the intersection is empty, return None.
        """
        intersection_offset = self.chunk_bbox.intersection(bbox) - self.chunk_bbox.start
        if intersection_offset.size == 0:
            return None
        return self.crop(intersection_offset)


FILENAME_PARTS = (
    'id',
    'mask_bbox',
    'size',
    'neighbors',
    'margin'
)
FILENAME_SEP = '__'
for p1 in FILENAME_PARTS:
    for p2 in FILENAME_PARTS:
        if p1 != p2:
            assert not p1.startswith(p2) and not p2.startswith(p1)


def _file_dir_encode(d: Direction) -> str:
    s = ''
    for i in d:
        if i == 0:
            s += '0'
        elif i < 0:
            s += 'n'
        else:
            s += 'p'
    return s


def _file_dir_decode(s: str) -> Direction:
    d = []
    for c in s:
        if c == '0':
            d.append(0)
        elif c == 'n':
            d.append(-1)
        elif c == 'p':
            d.append(1)
        else:
            raise ValueError(f'Invalid character in direction string: {c}')
    return Direction.from_collection(d)


def get_filename(
        seg_mask: SegmentMask,
        margin: Optional[int] = None,
        boundary_threshold: int = BOUNDARY_THRESHOLD,
) -> str:
    if margin is None:
        margin = seg_mask.margin
    neighbor_dirs = seg_mask.on_chunk_boundaries(margin=(margin * 2), threshold=boundary_threshold)
    neighbor_str = '+'.join([_file_dir_encode(d) for d in neighbor_dirs]) if neighbor_dirs else None
    folder = seg_mask.chunk_bbox.string
    parts = {
        'id': seg_mask.id,
        'mask_bbox': seg_mask.bbox.string,
        'size': seg_mask.size,
        'neighbors': neighbor_str,
        'margin': margin,
    }
    assert set(parts.keys()) == set(FILENAME_PARTS)
    return os.path.join(folder, FILENAME_SEP.join(f'{k}_{v}' for k, v in parts.items() if v is not None))


def mask_id_from_path(path: Union[str, Path]) -> str:
    return '/'.join(str(path).rsplit('/')[-2:])


def parse_filename(path: str) -> dict:
    folder, filename = str(path).rsplit('/')[-2:]
    parsed = {'chunk': folder}
    name, ext = os.path.splitext(filename)
    parts = name.split(FILENAME_SEP)
    assert all(any(p.startswith(k) for k in FILENAME_PARTS) for p in parts)
    remaining_keys = set(FILENAME_PARTS)
    for part in parts:
        for key in remaining_keys:
            if part.startswith(key):
                remaining_keys.remove(key)
                val = part.split(key + '_', maxsplit=1)[1]
                try:
                    parsed_val = int(val)
                except ValueError:
                    if key == 'neighbors':
                        parsed_val = [_file_dir_decode(d) for d in val.split('+')]
                        parsed[key] = parsed_val
                        break
                    else:
                        parsed_val = BoundingBox.from_string(val)
                if parsed_val is None or (isinstance(parsed_val, list) and None in parsed_val):
                    raise ValueError(f'Could not parse value for key {key} in filename {filename}: {val}')
                if isinstance(parsed_val, BoundingBox):
                    parsed_val = parsed_val.string
                elif isinstance(parsed_val, list):
                    parsed_val = [bbox.string for bbox in parsed_val]
                parsed[key] = parsed_val
                break
    for key in remaining_keys:
        parsed[key] = None

    # Handle finding neighbors from directions
    if parsed['neighbors'] is not None:
        chunk = BoundingBox.from_string(parsed['chunk'])
        neighbors = chunk.get_neighbors(directions=parsed['neighbors'], overlap=(2 * parsed.get('margin', 0)))
        parsed['neighbors'] = [neighbor.string for neighbor in neighbors]

    return parsed


def _get_seg_array(
        chunk: Chunk,
        segmentation_params: SegmentationParams,
) -> np.ndarray:
    if not chunk.is_segmentation and segmentation_params.threshold is None:
        raise ValueError('The input chunk is not a segmentation, so a threshold must be provided.')
    assert chunk.array.ndim == 3

    if segmentation_params.threshold is not None:
        seg_array = threshold_segmentation(chunk.array, segmentation_params.threshold)
    else:
        seg_array = chunk.array.copy()
    return seg_array


def _get_seg_sizes_by_id(
        seg_array: np.ndarray,
        segmentation_params: SegmentationParams,
) -> Dict[int, int]:
    seg_sizes_by_id = dict(zip(*fastremap.unique(seg_array, return_counts=True)))
    seg_sizes_by_id.pop(0, None)
    total_count = len(seg_sizes_by_id)
    print(f'{total_count} segments found in the input chunk.', flush=True)
    if segmentation_params.min_size:
        seg_sizes_by_id = {k: v for k, v in seg_sizes_by_id.items() if v >= segmentation_params.min_size}
        if len(seg_sizes_by_id) < total_count:
            print(f'{len(seg_sizes_by_id)} segments remaining after filtering out those smaller than'
                  f' {segmentation_params.min_size}.', flush=True)
    return seg_sizes_by_id


def _save_segment_mask(
        mask: SegmentMask,
        dir_path: Union[str, Path],
        margin: Optional[int] = None,
        boundary_threshold: int = BOUNDARY_THRESHOLD,
):
    filename = get_filename(mask, margin=margin, boundary_threshold=boundary_threshold)
    path = Path(dir_path) / f'{filename}.h5'
    mask.save(path)


def _get_alternative_overlap(
        segment_mask: SegmentMask,
        alt_chunk_array: np.ndarray,
        alt_chunk_bbox: BoundingBox,
        detection_threshold: float,
        min_value: Optional[float] = None,
) -> Optional[AltOverlapInfo]:
    """
    Check if segment is actually a misclassified RNA/RNP granule.
    """
    overlap_bbox = alt_chunk_bbox.intersection(segment_mask.bbox)
    alternative_bbox = overlap_bbox - alt_chunk_bbox.start
    overlap_array = alt_chunk_array[*alternative_bbox.slices]
    mask_bbox = overlap_bbox - segment_mask.bbox.start
    mask_array = segment_mask.mask[*mask_bbox.slices]
    overlap_mask = _apply_threshold(overlap_array * mask_array, detection_threshold).astype(np.bool_)
    overlap_count = overlap_mask.sum()
    overlap_fraction = overlap_count / segment_mask.size

    if overlap_count == 0:
        return None
    else:
        overlap_max = overlap_array[overlap_mask].max()
        if min_value and overlap_max < min_value:
            return None
        else:
            params = SegmentationParams(threshold=detection_threshold, min_value=min_value)
            return AltOverlapInfo(overlap_count, overlap_fraction, overlap_max, params)


def _get_neu_segment_overlaps(
        segment_mask: SegmentMask,
        neu_array: np.ndarray,
        neu_bbox: BoundingBox,
) -> Optional[Dict[int, float]]:
    """
    Get the overlaps of the segment mask with the given neuron segmentation.
    """
    overlap_bbox = neu_bbox.intersection(segment_mask.bbox)
    neu_seg_bbox = overlap_bbox - neu_bbox.start
    overlap_array = neu_array[*neu_seg_bbox.slices]
    if overlap_array.size == 0:
        return None
    mask_bbox = overlap_bbox - segment_mask.bbox.start
    mask_array = segment_mask.mask[*mask_bbox.slices]
    neu_overlaps_by_id = dict(zip(*fastremap.unique(overlap_array[mask_array], return_counts=True)))
    neu_overlaps_by_id.pop(0, None)
    if neu_overlaps_by_id:
        return {k: (v / segment_mask.size)
                for k, v in sorted(neu_overlaps_by_id.items(), key=lambda x: x[1], reverse=True)}
    else:
        return None


@dataclass
class ProcessMaskData:
    seg_array: Any
    seg_id: Any
    chunk_array: Any
    chunk_bbox: Any
    seg_params: Any
    alt_chunk_array: Any
    alt_chunk_bbox: Any
    alt_seg_params: Any
    neu_chunk_array: Any
    neu_chunk_bbox: Any
    margin: Any
    boundary_threshold: Any
    save_dir: Any

    def to_dict(self) -> dict:
        d = {
            'seg_array': self.seg_array,
            'seg_id': self.seg_id,
            'chunk_array': self.chunk_array,
            'chunk_bbox': self.chunk_bbox.string,
            'seg_params': asdict(self.seg_params),
            'alt_chunk_array': self.alt_chunk_array,
            'alt_chunk_bbox': self.alt_chunk_bbox.string,
            'alt_seg_params': asdict(self.alt_seg_params),
            'neu_chunk_array': self.neu_chunk_array,
            'neu_chunk_bbox': self.neu_chunk_bbox.string,
            'margin': self.margin,
            'boundary_threshold': self.boundary_threshold,
            'save_dir': self.save_dir,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = d.copy()
        d['chunk_bbox'] = BoundingBox.from_string(d['chunk_bbox'])
        d['alt_chunk_bbox'] = BoundingBox.from_string(d['alt_chunk_bbox'])
        d['neu_chunk_bbox'] = BoundingBox.from_string(d['neu_chunk_bbox'])
        d['seg_params'] = SegmentationParams(**d['seg_params'])
        d['alt_seg_params'] = SegmentationParams(**d['alt_seg_params'])
        return cls(**d)

    def iter_over_ids(self) -> Iterator[dict]:
        for seg_id in self.seg_id:
            d = self.to_dict()
            d['seg_id'] = seg_id
            yield d


def _process_mask(data: dict) -> str:
    data = ProcessMaskData.from_dict(data)

    if isinstance(data.seg_array, SharedMemoryContainer):
        seg_array_shm = data.seg_array
        seg_array = seg_array_shm.load()
    else:
        seg_array = data.seg_array
        seg_array_shm = None

    if isinstance(data.chunk_array, SharedMemoryContainer):
        chunk_array_shm = data.chunk_array
        chunk_array = chunk_array_shm.load()
    else:
        chunk_array = data.chunk_array
        chunk_array_shm = None

    if isinstance(data.alt_chunk_array, SharedMemoryContainer):
        alt_chunk_array_shm = data.alt_chunk_array
        alt_chunk_array = alt_chunk_array_shm.load()
    else:
        alt_chunk_array = data.alt_chunk_array
        alt_chunk_array_shm = None

    if isinstance(data.neu_chunk_array, SharedMemoryContainer):
        neu_chunk_array_shm = data.neu_chunk_array
        neu_chunk_array = neu_chunk_array_shm.load()
    else:
        neu_chunk_array = data.neu_chunk_array
        neu_chunk_array_shm = None

    mask = (seg_array == data.seg_id)
    size = mask.sum()
    if size == 0:
        return 'size 0'

    bbox_in_chunk = BoundingBox.from_points(np.argwhere(mask))
    mask = mask[*bbox_in_chunk.slices]
    bbox = bbox_in_chunk + data.chunk_bbox.start
    mask = SegmentMask(
        id=data.seg_id,
        size=size,
        bbox=bbox,
        bbox_in_chunk=bbox_in_chunk,
        chunk_bbox=data.chunk_bbox,
        margin=data.margin,
        mask=mask,
    )

    max_val = chunk_array[mask.chunk_mask].max()
    if data.seg_params.min_value is not None and (max_val < data.seg_params.min_value):
        return f'min value ({max_val}) < {data.seg_params.min_value}'

    if data.margin:
        mask_inside_margin = mask.crop(data.margin)
        if mask_inside_margin is None:
            return 'outside margin'

    neu_overlaps = _get_neu_segment_overlaps(mask, neu_chunk_array, data.neu_chunk_bbox)
    mask.neu_seg_overlaps = neu_overlaps

    alt_overlap = _get_alternative_overlap(
        mask, alt_chunk_array, data.alt_chunk_bbox, data.alt_seg_params.threshold, data.alt_seg_params.min_value)
    mask.alt_overlap_info = alt_overlap

    _save_segment_mask(mask, data.save_dir, margin=data.margin, boundary_threshold=data.boundary_threshold)
    return 'saved'


def _process_masks(
        mask_chunk: Chunk,
        seg_params: SegmentationParams,
        alt_chunk: Chunk,
        alt_seg_params: SegmentationParams,
        neu_seg_chunk: Chunk,
        margin: int,
        boundary_threshold: int,
        save_dir: Union[str, Path],
        workers: int = 1,
        force_multiprocessing: bool = False,
) -> None:
    chunk_array = mask_chunk.array
    chunk_bbox = mask_chunk.bbox
    alt_chunk_array = alt_chunk.array
    alt_chunk_bbox = alt_chunk.bbox
    neu_chunk_array = neu_seg_chunk.array
    neu_chunk_bbox = neu_seg_chunk.bbox

    seg_array = _get_seg_array(mask_chunk, seg_params)
    seg_sizes_by_id = _get_seg_sizes_by_id(seg_array, seg_params)

    if workers:
        if workers < 0:
            workers = multiprocessing.cpu_count() + workers
            if workers <= 0:
                workers = 1
        workers = min(workers, multiprocessing.cpu_count(), len(seg_sizes_by_id))
    else:
        workers = 1

    if workers > 1 or force_multiprocessing:
        with SharedMemoryManager() as smm:
            seg_array_shm = SharedMemoryContainer.create(seg_array)
            chunk_array_shm = SharedMemoryContainer.create(chunk_array)
            alt_chunk_array_shm = SharedMemoryContainer.create(alt_chunk_array)
            neu_chunk_array_shm = SharedMemoryContainer.create(neu_chunk_array)
            data = (
                ProcessMaskData(
                    seg_array_shm,
                    seg_id,
                    chunk_array_shm,
                    chunk_bbox,
                    seg_params,
                    alt_chunk_array_shm,
                    alt_chunk_bbox,
                    alt_seg_params,
                    neu_chunk_array_shm,
                    neu_chunk_bbox,
                    margin,
                    boundary_threshold,
                    save_dir
                ).to_dict()
                for seg_id in seg_sizes_by_id.keys()
            )
            with multiprocessing.Pool(workers) as pool:
                results = pool.map(_process_mask, data)
            seg_array_shm.close()
            seg_array_shm.unlink()
            chunk_array_shm.close()
            chunk_array_shm.unlink()
            alt_chunk_array_shm.close()
            alt_chunk_array_shm.unlink()
            neu_chunk_array_shm.close()
            neu_chunk_array_shm.unlink()
    else:
        data = ProcessMaskData(
            seg_array,
            list(seg_sizes_by_id.keys()),
            chunk_array,
            chunk_bbox,
            seg_params,
            alt_chunk_array,
            alt_chunk_bbox,
            alt_seg_params,
            neu_chunk_array,
            neu_chunk_bbox,
            margin,
            boundary_threshold,
            save_dir,
        )
        results = list(map(_process_mask, tqdm(data.iter_over_ids())))

    reasons = defaultdict(int)
    for r in results:
        if r.startswith('min value'):
            r = 'min value'
        reasons[r] += 1
    print(f"{reasons.pop('saved', 0)} segments saved to {save_dir} after processing.", flush=True)
    print(f"Reasons for not saving: {', '.join([f'{k} ({v})' for k, v in reasons.items()])}", flush=True)
    return None


def run_save_chunk_segmentation(
        mito_chunk: Chunk,
        granule_chunk: Chunk,
        neu_seg_chunk: Chunk,
        mito_dir: Union[str, Path],
        granule_dir: Union[str, Path],
        margin: int,
        mito_proba_threshold=DEFAULT_MITO_SEG_PARAMS.threshold,
        mito_min_size=DEFAULT_MITO_SEG_PARAMS.min_size,
        mito_min_value=DEFAULT_MITO_SEG_PARAMS.min_value,
        granule_proba_threshold=DEFAULT_GRANULE_SEG_PARAMS.threshold,
        granule_min_size=DEFAULT_GRANULE_SEG_PARAMS.min_size,
        granule_min_value=DEFAULT_GRANULE_SEG_PARAMS.min_value,
        boundary_threshold: int = BOUNDARY_THRESHOLD,
        workers: int = 1,
        force_multiprocessing: bool = False,
) -> None:

    mito_chunk = _as_proba(mito_chunk)
    granule_chunk = _as_proba(granule_chunk)

    mito_seg_params = SegmentationParams(mito_proba_threshold, mito_min_size, mito_min_value)
    granule_seg_params = SegmentationParams(granule_proba_threshold, granule_min_size, granule_min_value)

    _process_masks(mito_chunk, mito_seg_params, granule_chunk, granule_seg_params, neu_seg_chunk,
                   margin, boundary_threshold, mito_dir, workers, force_multiprocessing)
    _process_masks(granule_chunk, granule_seg_params, mito_chunk, mito_seg_params, neu_seg_chunk,
                   margin, boundary_threshold, granule_dir, workers, force_multiprocessing)


def label_segments(
        bbox: BoundingBox,
        seg_masks_dir: Union[str, Path],
        seg_id_map: Optional[Union[str, Dict[str, int]]] = None,
        expand_margin: int = 0,
        crop_margin: int = 0,
        voxel_size: Cartesian = Cartesian(8, 8, 8),
) -> Chunk:
    if expand_margin and crop_margin:
        raise ValueError('expand_margin and crop_margin cannot be used together.')
    if expand_margin:
        overlapping_bbox = bbox.adjust(expand_margin)
        chunk_bbox = bbox.copy()
    elif crop_margin:
        overlapping_bbox = bbox.copy()
        chunk_bbox = bbox.adjust(-crop_margin)
    else:
        overlapping_bbox = bbox.copy()
        chunk_bbox = bbox.copy()

    if isinstance(seg_masks_dir, str):
        seg_masks_dir = Path(seg_masks_dir)
    seg_mask_pattern = str(seg_masks_dir / overlapping_bbox.string / '*.h5')
    seg_mask_paths = glob(seg_mask_pattern)
    print(f'{len(seg_mask_paths)} segmentation mask files found matching {seg_mask_pattern}')

    if seg_id_map is None:
        seg_id_map = seg_masks_dir / 'mask_to_seg_map.json'
    if not isinstance(seg_id_map, dict):
        seg_id_map = str(seg_id_map)
    if isinstance(seg_id_map, str):
        if not os.path.exists(seg_id_map) and str(seg_masks_dir) not in seg_id_map:
            seg_id_map = seg_masks_dir / seg_id_map
        if not os.path.exists(seg_id_map):
            raise ValueError(f'seg_id_map file not found: {seg_id_map}')
        with open(seg_id_map) as f:
            seg_id_map = json.load(f)

    seg_array = np.zeros(chunk_bbox.shape, dtype=np.uint64)
    if 'DISBATCH_REPEAT_INDEX' not in os.environ:
        mask_path_iter = tqdm(seg_mask_paths)
    else:
        mask_path_iter = seg_mask_paths

    count = 0
    for path in mask_path_iter:
        seg_id = seg_id_map.get(mask_id_from_path(path))
        if seg_id:
            seg_mask = SegmentMask.load(path)
            mask_array = seg_mask.intersection(chunk_bbox).chunk_mask
            seg_array[mask_array] = np.uint64(seg_id)
            count += 1
    print(f'{count} segment masks labeled.')

    return Chunk(seg_array, voxel_offset=chunk_bbox.start, voxel_size=voxel_size, layer_type='segmentation')


"""
Procedure:
- expand chunk margin
- find and save all segments using generate_segment_masks and _save_segment_mask
  > segment chunk, neighbor chunks, and mask bbox are specified in filename
- cycle through all saved segment masks
  > if segment is on a chunk boundary, filter candidate overlaps by looking at
    segments in neighbor chunk(s) that are on the boundary with this chunk
  > if these overlap candidates have overlapping mask bboxes, load the masks
    within the overlap region and get the overlapping voxel count
  > if the overlapping voxel count is above a threshold, save the pairs for
    merging
- consolidate and merge all segments
  > create merge map: {segment_id: [segment_id]}
  > for each segment pair to merge, update the merge map with the partnered segment IDs,
    keeping the joined segment merges in sync across the map
  > create a relabel map with consecutive segment IDs
- once all merges have been catalogued, create a new set of segment mask files:
  > if no merges, copy the original segment mask file to the new directory with the new ID from the relabel map
  > if there are joined segments, merge the segment masks and save the new mask with the new ID from the relabel map
"""
