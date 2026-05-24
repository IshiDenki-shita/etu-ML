from experiments.segmentation.context.segmentation_context import SegmentationContext
from experiments.segmentation.selector.selector_base import SelectorBase


class DPSelector(SelectorBase):
    def process(self, context: SegmentationContext) -> SegmentationContext:
        if context.selected_boundaries is None:
            context.selected_boundaries = []
        return context
