from datetime import datetime
from collections.abc import Iterable

import numpy as np

from cell_tracking.types_ import ZarrChunk
from cell_tracking.reader import ZarrReader

def test_zarr_reader():
    reader = ZarrReader(path='data/train/44b6_f28707c6.zarr')
    assert isinstance(reader, Iterable)

    for timestamp, chunk in reader:
        # test the iterand
        assert isinstance(chunk, ZarrChunk)
        assert isinstance(timestamp, datetime)

        # test the internals of the iterand
        assert isinstance(chunk.voxels, np.ndarray)
        assert isinstance(chunk.scale, np.ndarray)
        assert isinstance(chunk.timestamp, datetime) or (chunk.timestamp is None)

        break