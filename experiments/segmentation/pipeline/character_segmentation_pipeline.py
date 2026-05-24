import numpy as np

from experiments.segmentation.candidate.graph_candidate_generator import (
    GraphCandidateGenerator,
)
from experiments.segmentation.context.segmentation_context import SegmentationContext
from experiments.segmentation.pipeline.line_removal import LineRemover
from experiments.segmentation.preprocessing.preprocessor import preprocess_image
from experiments.segmentation.selector.dp_selector import DPSelector
from experiments.segmentation.valley.gradient_valley_detector import (
    GradientValleyDetector,
)


class PreprocessingStep:
    def __init__(self, binary_threshold: int) -> None:
        self._binary_threshold = binary_threshold

    def process(self, context: SegmentationContext) -> SegmentationContext:
        context.binary = preprocess_image(
            context.original_image, self._binary_threshold
        )
        return context


class LineRemovalStep:
    def __init__(self) -> None:
        self._line_remover = LineRemover()

    def process(self, context: SegmentationContext) -> SegmentationContext:
        context.removed_binary = self._line_remover.remove_lines(
            img=context.original_image,
            visualize=False,
        )
        return context


class ValleyDetectionStep:
    def __init__(self, min_theta: float) -> None:
        self._valley_detector = GradientValleyDetector(min_theta)

    def process(self, context: SegmentationContext) -> SegmentationContext:
        return self._valley_detector.process(context)


class CandidateGenerationStep:
    def __init__(self) -> None:
        self._candidate_generator = GraphCandidateGenerator()

    def process(self, context: SegmentationContext) -> SegmentationContext:
        return self._candidate_generator.process(context)


class SelectorStep:
    def __init__(self) -> None:
        self._selector = DPSelector()

    def process(self, context: SegmentationContext) -> SegmentationContext:
        return self._selector.process(context)


class CropStep:
    def process(self, context: SegmentationContext) -> SegmentationContext:
        if context.character_images is None:
            context.character_images = []
        return context


class CharacterSegmentationPipeline:
    def __init__(self, config) -> None:
        self._steps = [
            PreprocessingStep(config.binary_threshold),
            LineRemovalStep(),
            ValleyDetectionStep(np.deg2rad(config.min_valley_theta_deg)),
            CandidateGenerationStep(),
            SelectorStep(),
            CropStep(),
        ]

    def process(self, context: SegmentationContext) -> SegmentationContext:
        for step in self._steps:
            context = step.process(context)
        return context
