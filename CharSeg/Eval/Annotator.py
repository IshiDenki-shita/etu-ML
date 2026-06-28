"""
A*候補(context.candidates)から正解の分割境界線を選ぶ
アノテーションツール。

パイプラインを Preprocesser → LineNoiseRemover → BlankTrimmer
→ AstarInterval まで実行し、生成された候補線の中から、
マウスクリックで「正解」をトグル選択して保存する。

操作方法:
    クリック  : 最も近い候補線を選択/選択解除（トグル）
    s キー    : 現在の選択を保存
    n キー    : 保存して次の画像へ
    q キー    : 保存して終了
    ウィンドウを閉じる: 保存して次の画像へ（nと同じ扱い）

実行方法:
    python -m CharSeg.Eval.Annotator
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import matplotlib.pyplot as plt
import numpy as np

from CharSeg.context import Context, Borderline
from CharSeg.preprocess import Preprocesser
from CharSeg.LineNoise.LineNoiseBold import LineNoiseRemover
from CharSeg.BlankTrim import BlankTrimmer
from CharSeg.GenCandidates.AstarInterval import AstarInterval

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff"}

UNSELECTED_COLOR = "#7EC8E3"
SELECTED_COLOR = "#E63946"


@dataclass
class AnnotatorConfig:
    input_dir: Path = Path("photos/sample/cells")
    output_path: Path = Path("CharSeg/Eval/annotations.json")
    # クリック位置がこの距離(px)より遠い候補にはヒットしないとみなす
    click_tolerance: float = 25.0


class Annotator:
    """A*候補から正解の境界線を選ぶインタラクティブなアノテーションツール。"""

    def __init__(self, config: AnnotatorConfig | None = None) -> None:
        self.cfg = config or AnnotatorConfig()

        self.preprocesser = Preprocesser()
        self.line_remover = LineNoiseRemover(debug=False)
        self.blank_trimmer = BlankTrimmer(debug=False)
        self.candidate_gen = AstarInterval(debug=False)

        self.annotations: dict = self._load_annotations()

        # 1枚分の作業状態（annotate_oneの間だけ使う）
        self._current_image_name: str = ""
        self._candidates: list[Borderline] = []
        self._selected: set[int] = set()
        self._lines: list = []  # matplotlib Line2D（候補と同じ順）
        self._fig = None
        self._ax = None
        self._exit_reason: Literal["next", "quit"] | None = None

        if os.name == "nt":
            plt.rcParams["font.family"] = "MS Gothic"
        else:
            plt.rcParams["font.family"] = "Hiragino Sans"

    # -------------------------
    # 入出力
    # -------------------------
    def _load_annotations(self) -> dict:
        if self.cfg.output_path.exists():
            with open(self.cfg.output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_annotations(self) -> None:
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.output_path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        logger.debug(f"アノテーションを保存しました: {self.cfg.output_path}")

    def _iter_image_paths(self) -> list[Path]:
        return sorted(
            p
            for p in self.cfg.input_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # -------------------------
    # パイプライン実行
    # -------------------------
    def _build_candidates(self, image_path: Path) -> Context:
        context = Context()
        context.image_path = image_path

        self.preprocesser.process(context)
        self.line_remover.process(context)
        self.blank_trimmer.process(context)
        self.candidate_gen.process(context)

        if context.candidates is None or context.blank_trimmed is None:
            raise ValueError("候補生成に失敗しました（contextが不完全です）。")

        return context

    # -------------------------
    # メインループ
    # -------------------------
    def run(self) -> None:
        image_paths = self._iter_image_paths()

        if not image_paths:
            logger.warning(f"画像が見つかりません: {self.cfg.input_dir}")
            return

        logger.debug(f"{len(image_paths)} 件の画像をアノテーションします")

        for image_path in image_paths:
            name = image_path.name
            prev = self.annotations.get(name)

            if prev is not None:
                logger.debug(f"{name} は既にアノテーション済みです。再編集します。")

            try:
                self.annotate_one(image_path, resume=prev)
            except Exception:
                logging.exception(f"アノテーション中にエラーが発生しました: {name}")
                continue

            if self._exit_reason == "quit":
                break

    def annotate_one(self, image_path: Path, resume: dict | None = None) -> None:
        context = self._build_candidates(image_path)
        candidates = context.candidates
        blank_trimmed = context.blank_trimmed

        assert candidates is not None and blank_trimmed is not None

        self._current_image_name = image_path.name
        self._candidates = candidates
        self._selected = self._restore_selection(image_path.name, candidates, resume)
        self._exit_reason = None

        self._fig, self._ax = plt.subplots(figsize=(9, 9))
        self._ax.imshow(blank_trimmed, cmap="gray")

        self._lines = []
        for cand in candidates:
            pts = np.asarray(cand, dtype=np.float64)
            (line,) = self._ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=UNSELECTED_COLOR,
                linewidth=1.2,
                alpha=0.5,
            )
            self._lines.append(line)

        self._refresh_view()

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.show()

        # ウィンドウを閉じた場合も含め、必ず保存する
        self._commit_current()

    def _restore_selection(
        self,
        name: str,
        candidates: list[Borderline],
        resume: dict | None,
    ) -> set[int]:
        if resume is None:
            return set()

        if resume.get("candidate_count") != len(candidates):
            logger.warning(
                f"{name}: 候補数が前回({resume.get('candidate_count')})と"
                f"異なるため({len(candidates)})、選択状態は復元しません。"
            )
            return set()

        return set(resume.get("selected_indices", []))

    # -------------------------
    # イベントハンドラ
    # -------------------------
    def _on_click(self, event) -> None:
        if event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return

        idx, dist = self._nearest_candidate(event.xdata, event.ydata)

        if idx is None or dist > self.cfg.click_tolerance:
            return

        self._toggle(idx)
        self._refresh_view()
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "s":
            self._commit_current()
        elif event.key == "n":
            self._exit_reason = "next"
            self._commit_current()
            plt.close(self._fig)
        elif event.key == "q":
            self._exit_reason = "quit"
            self._commit_current()
            plt.close(self._fig)

    # -------------------------
    # 補助
    # -------------------------
    def _toggle(self, idx: int) -> None:
        if idx in self._selected:
            self._selected.discard(idx)
        else:
            self._selected.add(idx)

    def _nearest_candidate(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_dist = float("inf")

        for i, cand in enumerate(self._candidates):
            pts = np.asarray(cand, dtype=np.float64)
            if len(pts) == 0:
                continue
            dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            d = float(dists.min())
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx, best_dist

    def _refresh_view(self) -> None:
        for i, line in enumerate(self._lines):
            if i in self._selected:
                line.set_color(SELECTED_COLOR)
                line.set_linewidth(2.4)
                line.set_alpha(1.0)
            else:
                line.set_color(UNSELECTED_COLOR)
                line.set_linewidth(1.2)
                line.set_alpha(0.5)

        if self._ax is not None:
            self._ax.set_title(self._title_text())

    def _title_text(self) -> str:
        return (
            f"{self._current_image_name}  |  "
            f"selected {len(self._selected)}/{len(self._candidates)}  |  "
            f"click=toggle  s=save  n=next  q=quit"
        )

    def _commit_current(self) -> None:
        self.annotations[self._current_image_name] = self._make_record()
        self._save_annotations()

    def _make_record(self) -> dict:
        indices = sorted(self._selected)
        rep_xs = [self._representative_x(self._candidates[i]) for i in indices]
        return {
            "candidate_count": len(self._candidates),
            "selected_indices": indices,
            "representative_xs": rep_xs,
        }

    def _representative_x(self, borderline: Borderline) -> float:
        ys = [p[1] for p in borderline]
        y_center = (min(ys) + max(ys)) / 2
        closest_point = min(borderline, key=lambda p: abs(p[1] - y_center))
        return float(closest_point[0])


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    annotator = Annotator()
    annotator.run()


if __name__ == "__main__":
    main()
