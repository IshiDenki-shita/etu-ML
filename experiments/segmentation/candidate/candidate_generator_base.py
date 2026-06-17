from abc import ABC, abstractmethod

from experiments.segmentation.context.segmentation_context import SegmentationContext


class CandidateGeneratorBase(ABC):
    @abstractmethod
    def process(self, context: SegmentationContext) -> SegmentationContext:
        pass
