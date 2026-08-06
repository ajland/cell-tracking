from datetime import datetime, timedelta

import numpy as np
import pytest

from cell_tracking.types_ import ZarrChunk
from cell_tracking.detector import BinaryChunkDetector

class DummyFeeder:
    def __init__(self, *frames):
        self.frames = frames

    def __iter__(self):
        t = datetime(2024, 1, 1)

        for i, frame in enumerate(self.frames):
            timestamp = t + timedelta(seconds=i)
            chunk = ZarrChunk(voxels=frame, scale=np.zeros(3), timestamp=timestamp)
            yield timestamp, chunk

@pytest.mark.parametrize(
    "image, expected_count",
    [
        (np.zeros((5,5,5), bool), 0),
        (np.array([np.eye(5, dtype=bool), np.eye(5, dtype=bool)]), 5),
        (np.ones((5,5,5), bool), 1),
        (np.array([np.ones((5,5)), np.zeros((5,5)), np.ones((5,5))]), 2)
    ],
)
def test_number_of_detections(image, expected_count):

    detector = BinaryChunkDetector(DummyFeeder(image))

    _, detections = next(iter(detector))

    assert len(detections) == expected_count