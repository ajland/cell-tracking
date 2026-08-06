from collections.abc import Callable

import numpy as np
from scipy.ndimage import label, uniform_filter
from stonesoup.base import Property
from stonesoup.feeder.base import Feeder
from stonesoup.buffered_generator import BufferedGenerator

class SmoothThresholder(Feeder):
    """ZarrChunk feeder
    
    Smooth the ZarrChunk.voxels according to a filter and apply a
    threshold to create a binary mask.
    """
    filter_func: Callable = Property(
        doc = "A filter which takes only a single np.ndarray as input",
        default=lambda input: uniform_filter(input.astype(np.float32), size=3)
    )
    percentile: float = Property(doc="Determines binary mask threshold")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for timestamp, chunk in self.reader:
            smoothed = self.filter_func(chunk.voxels)
            threshold = threshold = np.percentile(smoothed, self.percentile)
            chunk.voxels = smoothed > threshold
            yield timestamp, chunk