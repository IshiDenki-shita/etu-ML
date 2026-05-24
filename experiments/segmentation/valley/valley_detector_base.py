from abc import ABC, abstractmethod

from experiments.segmentation.context.segmentation_context import SegmentationContext


class ValleyDetectorBase(ABC):
    @abstractmethod
    def process(self, context: SegmentationContext) -> SegmentationContext:
        pass
