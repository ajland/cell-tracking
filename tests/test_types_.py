import datetime

import numpy as np
import pytest

from cell_tracking.types_ import ZarrChunk

def test_zarr_chunk():
    chunk_np = np.random.random((15, 15))
    scale_np = np.random.random(3)
    timestamp = datetime.datetime.now()

    # No timestamp
    volume = ZarrChunk(chunk_np, scale_np)
    np.testing.assert_array_equal(volume.voxels, chunk_np)
    np.testing.assert_array_equal(volume.scale, scale_np)
    assert volume.timestamp is None

    # With timestamp
    volume2 = ZarrChunk(chunk_np, scale_np, timestamp)
    np.testing.assert_array_equal(volume.voxels, volume2.voxels)
    np.testing.assert_array_equal(volume.scale, volume2.scale)
    assert volume2.timestamp == timestamp

    # Expect Type error
    with pytest.raises(TypeError):
        ZarrChunk()