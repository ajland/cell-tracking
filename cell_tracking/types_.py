from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

class ZarrChunk(SensorData):
    """ Zarr Chunk type used to represent a single volume chunk from a Zarr directory """

    voxels: np.ndarray = Property(
        doc="An array of shape (Z,Y,X) containing the individual voxel values, where Z,Y,"
        "X are the three physical dimensions of the 3D volume.")
    scale: np.ndarray = Property(
        doc="An array of length three, ordered z,y,x. This is the physical scale" \
        "of the voxel in each of the three dimensions."
    )
    timestamp: datetime = Property(doc="An optional timestamp", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.voxels.shape) != len(self.scale):
            raise ValueError("voxels has different dimensionalty than scale")

    def __bool__(self):
        return len(self.voxels) > 0