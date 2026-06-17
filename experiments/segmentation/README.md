# Segmentation (experiments/segmentation)

This README lists common ways to run the segmentation pipeline and to visualize intermediate results.

Prerequisites
- Use the project's virtual environment (example path below). Ensure dependencies from `requirements.txt` are installed.

Quick checks
```bash
# show python in venv
.venv/bin/python -V
```

Run pipeline (no GUI)
```bash
MPLBACKEND=Agg .venv/bin/python -m experiments.segmentation.main
```

Run pipeline and show debug visualization in same process
```bash
.venv/bin/python -c "from experiments.segmentation.pipeline.segmenter import CharacterSegmentationConfig, CharacterSegmenter; from experiments.segmentation.visualization.debug_visualizer import visualize_context; s=CharacterSegmenter(CharacterSegmentationConfig()); s.run(); visualize_context(s.context)"
```

Run pipeline programmatically then visualize later
```bash
# 1) run pipeline and pickle context (example script)
# 2) load context and call visualize_context(context)
```

REPL / step debugging examples
```bash
.venv/bin/python -c "from experiments.segmentation.pipeline.character_segmentation_pipeline import PreprocessingStep; from experiments.segmentation.context.segmentation_context import SegmentationContext; import cv2; img=cv2.imread('photos/sample/cells/ebiten.jpeg'); ctx=SegmentationContext(original_image=img); ctx=PreprocessingStep(0).process(ctx); print(type(ctx.binary), hasattr(ctx,'binary'))"
```

Files of interest
- Entry: `experiments/segmentation/main.py`
- Pipeline orchestrator: `experiments/segmentation/pipeline/character_segmentation_pipeline.py`
- Runner wrapper: `experiments/segmentation/pipeline/segmenter.py`
- Context: `experiments/segmentation/context/segmentation_context.py`
- Visualization (debug-only): `experiments/segmentation/visualization/debug_visualizer.py`

Notes
- The pipeline does not call visualization directly; use `visualize_context(context)` to render figures.
- Algorithm modules no longer import matplotlib or call `cv2.imshow`.
