import os
import json
import datetime
from collections.abc import Iterator

import blosc2
import numpy as np
from stonesoup.reader.file import FileReader
from stonesoup.reader.base import FrameReader
from stonesoup.buffered_generator import BufferedGenerator

from .types_ import ZarrChunk

class ZarrReader(FileReader, FrameReader):
    '''ZarrReader reads volumes from zarr directory'''

    @BufferedGenerator.generator_method
    def frames_gen(self) -> Iterator[tuple[datetime.datetime, set[ZarrChunk]]]:
        """Returns a generator of frames for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step

        expanded from: 
        https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor
        """
        with open(os.path.join(self.path, 'zarr.json')) as f:
            zarr_meta = json.load(f)
        scale = zarr_meta['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        time_scale = scale[0]
        scale = np.array(scale[1:])

        with open(os.path.join(self.path, '0', 'zarr.json')) as f:
            arr_meta = json.load(f)
        shape = tuple(arr_meta['shape'])  # (T, Z, Y, X)
        dtype = np.dtype(arr_meta['data_type'])
        n_t = shape[0]

        # arbitrary start time set to January 1, 2026 00:00:00
        start_time = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

        for t in range(n_t):
            chunk_path = os.path.join(self.path, '0', 'c', str(t), '0', '0', '0')
            with open(chunk_path, 'rb') as f:
                compressed = f.read()
            decompressed = blosc2.decompress(compressed)
            vol = np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])  # (Z, Y, X)

            # ds = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
            timestamp = start_time + datetime.timedelta(seconds=t*time_scale)
            yield timestamp, ZarrChunk(voxels=vol, scale=scale, timestamp=timestamp)

if __name__ == '__main__':
    reader = ZarrReader(path='data/dense_channel/train/44b6_f28707c6.zarr')

    for volume in reader:
        continue