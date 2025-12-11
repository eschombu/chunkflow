from scipy.ndimage import gaussian_filter

from chunkflow.chunk import Chunk


def execute(chunk: Chunk, sigma: float=1., inplace=False):
    if not inplace:
        chunk = chunk.clone()

    if chunk.ndim == 4:
        for channel in range(chunk.shape[0]):
            chunk.array[channel,:,:,:] = gaussian_filter(chunk.array[channel,:,:,:], sigma=sigma)
    elif chunk.ndim == 3:
        chunk.array = gaussian_filter(chunk.array, sigma=sigma)
    else:
        raise ValueError(f'only support 4 or 3d, but got {chunk.ndim}')

    return chunk
