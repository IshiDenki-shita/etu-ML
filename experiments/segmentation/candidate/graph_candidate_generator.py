from experiments.segmentation.candidate.candidate_generator_base import (
    CandidateGeneratorBase,
)
from experiments.segmentation.context.segmentation_context import SegmentationContext


class GraphCandidateGenerator(CandidateGeneratorBase):
    def process(self, context: SegmentationContext) -> SegmentationContext:
        if context.candidate_boundaries is None:
            context.candidate_boundaries = []
        return context
