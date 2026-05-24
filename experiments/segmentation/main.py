from experiments.segmentation.pipeline.segmenter import (
    CharacterSegmentationConfig,
    CharacterSegmenter,
)


def main() -> None:
    config = CharacterSegmentationConfig()
    segmenter = CharacterSegmenter(config)
    segmenter.run()


if __name__ == "__main__":
    main()
