import datetime
from collections.abc import Iterator

import numpy as np
from scipy.ndimage import label
from ordered_set import OrderedSet
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.detection import Detection
from stonesoup.detector import Detector
from stonesoup.models.measurement import MeasurementModel

class BinaryChunkDetector(Detector):
    measurement_model: MeasurementModel = Property(
        doc="Measurement model used to generate detections",
        default=None
    )

    @BufferedGenerator.generator_method
    def detections_gen(self) -> Iterator[tuple[datetime.datetime, OrderedSet[Detection]]]:
        """Returns a generator of detections for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Detection`
            Detections generate in the time step
        """
        node_id_counter = 1
        t = 0
        for timestamp, chunk in self.sensor:
            detections = OrderedSet()
            labeled, n_features = label(chunk.voxels)

            for comp_id in range(1, n_features + 1):
                coords = np.argwhere(labeled == comp_id)
                centroid = coords.mean(axis=0)
                nid = node_id_counter
                node_id_counter += 1
                metadata = {
                    'row_type': 'node',
                    'node_id': nid,
                    't': t,
                    'z': int(centroid[0]),
                    'y': int(centroid[1]),
                    'x': int(centroid[2]),
                    'source_id': -1,
                    'target_id': -1,
                }
                detections.add(
                    Detection(state_vector=centroid,
                              timestamp=timestamp,
                              measurement_model=self.measurement_model,
                              metadata=metadata)
                )
                t+=1
                
            yield timestamp, detections