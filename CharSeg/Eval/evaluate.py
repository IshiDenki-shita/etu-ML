"""
DPselectorの選択結果と、Annotator(CharSeg/Eval/Annotator.py)で作成した
正解アノテーション(annotations.json)を比較する評価スクリプト。

評価方法:
    DPselectorの候補リストはAnnotatorの候補リストと先頭にleft_edge候補
    が追加されている分インデックスがずれるため、インデックスではなく
    representative_x(境界線のx座標)で最近傍マッチングして比較する。
    マッチングは scipy.optimize.linear_sum_assignment による
    最小コスト1対1割当で行い、距離が許容誤差を超えるペアは不採用とする。

実行方法:
    python -m CharSeg.Eval.evaluate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from CharSeg.context import Context, Borderline
from CharSeg.preprocess import Preprocesser
from CharSeg.LineNoise.LineNoiseBold import LineNoiseRemover
from CharSeg.BlankTrim import BlankTrimmer
from CharSeg.GenCandidates.AstarInterval import AstarInterval
from CharSeg.DPselector import DPselector, DPselectorConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    input_dir: Path = Path("photos/sample/cells")
    annotations_path: Path = Path("CharSeg/Eval/annotations.json")
    # マッチング許容誤差 = min_char_width * この比率（画像ごとに動的決定）
    tolerance_ratio_of_min_char_width: float = 0.5


@dataclass
class ImageEvalResult:
    image_name: str
    n_truth: int
    n_predicted: int
    n_matched: int
    n_missed: int
    n_extra: int
    matched_truth_xs: list[float] = field(default_factory=list)
    matched_pred_xs: list[float] = field(default_factory=list)

    @property
    def mean_abs_error(self) -> float | None:
        if not self.matched_truth_xs:
            return None
        errors = [
            abs(t - p) for t, p in zip(self.matched_truth_xs, self.matched_pred_xs)
        ]
        return float(np.mean(errors))


def representative_x(borderline: Borderline) -> float:
    """DPselector._representative_x と同一のロジック（依存を増やさないため複製）。"""
    ys = [p[1] for p in borderline]
    y_center = (min(ys) + max(ys)) / 2
    closest_point = min(borderline, key=lambda p: abs(p[1] - y_center))
    return float(closest_point[0])


def match_xs(
    truth_xs: list[float], pred_xs: list[float], tolerance: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    truth_xsとpred_xsを最小コスト割当(linear_sum_assignment)で1対1マッチングする。
    距離がtoleranceを超えるペアはマッチとして採用しない。

    戻り値:
        matched_pairs: [(truth_idx, pred_idx), ...]
        missed_truth_indices: マッチしなかった正解のindex一覧
        extra_pred_indices: マッチしなかった選択結果のindex一覧
    """
    if not truth_xs or not pred_xs:
        return [], list(range(len(truth_xs))), list(range(len(pred_xs)))

    cost = np.abs(np.subtract.outer(truth_xs, pred_xs))  # (n_truth, n_pred)
    truth_idx, pred_idx = linear_sum_assignment(cost)

    matched_pairs: list[tuple[int, int]] = []
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()

    for ti, pi in zip(truth_idx, pred_idx):
        if cost[ti, pi] <= tolerance:
            matched_pairs.append((int(ti), int(pi)))
            matched_truth.add(int(ti))
            matched_pred.add(int(pi))

    missed = [i for i in range(len(truth_xs)) if i not in matched_truth]
    extra = [i for i in range(len(pred_xs)) if i not in matched_pred]

    return matched_pairs, missed, extra


class Evaluator:
    def __init__(self, config: EvalConfig | None = None) -> None:
        self.cfg = config or EvalConfig()

        self.preprocesser = Preprocesser()
        self.line_remover = LineNoiseRemover(debug=False)
        self.blank_trimmer = BlankTrimmer(debug=False)
        self.candidate_gen = AstarInterval(debug=False)

        # 画像名 -> Context (candidates生成済み、DPselector未実行)
        self._context_cache: dict[str, Context] = {}
        self._annotations_cache: dict | None = None

    def _load_annotations(self) -> dict:
        if self._annotations_cache is None:
            with open(self.cfg.annotations_path, "r", encoding="utf-8") as f:
                self._annotations_cache = json.load(f)
        return self._annotations_cache

    def all_image_names(self) -> list[str]:
        """annotations.jsonに含まれる全画像名のリスト(train/test分割用)。"""
        return list(self._load_annotations().keys())

    def _build_or_get_context(self, image_name: str, image_path: Path) -> Context:
        """
        A*候補生成(Preprocesser~AstarInterval)はDPselectorのパラメータに
        依存しないため、パラメータ探索の対象外。画像ごとに1回だけ実行して
        キャッシュし、グリッドサーチ時の重複計算を避ける。
        """
        if image_name in self._context_cache:
            return self._context_cache[image_name]

        context = Context()
        context.image_path = image_path

        self.preprocesser.process(context)
        self.line_remover.process(context)
        self.blank_trimmer.process(context)
        self.candidate_gen.process(context)

        self._context_cache[image_name] = context
        return context

    def evaluate_one(
        self,
        image_name: str,
        record: dict,
        dp_config: DPselectorConfig | None = None,
    ) -> ImageEvalResult | None:
        image_path = self.cfg.input_dir / image_name
        if not image_path.exists():
            logger.warning(f"画像が見つかりません: {image_path}")
            return None

        context = self._build_or_get_context(image_name, image_path)

        if context.candidates is None or context.blank_trimmed is None:
            logger.warning(f"{image_name}: 候補生成に失敗しています。")
            return None

        dpselector = DPselector(debug=False, cfg=dp_config)
        dpselector.process(context)

        if context.selected is None:
            logger.warning(f"{image_name}: DPselectorの出力が不正です。")
            return None

        truth_xs = record["representative_xs"]
        pred_xs = [representative_x(b) for b in context.selected]

        _, min_char_width = dpselector.adopt_char_width(context.blank_trimmed)
        tolerance = self.cfg.tolerance_ratio_of_min_char_width * min_char_width

        matched_pairs, missed, extra = match_xs(truth_xs, pred_xs, tolerance)

        return ImageEvalResult(
            image_name=image_name,
            n_truth=len(truth_xs),
            n_predicted=len(pred_xs),
            n_matched=len(matched_pairs),
            n_missed=len(missed),
            n_extra=len(extra),
            matched_truth_xs=[truth_xs[ti] for ti, _ in matched_pairs],
            matched_pred_xs=[pred_xs[pi] for _, pi in matched_pairs],
        )

    def evaluate_all(
        self,
        dp_config: DPselectorConfig | None = None,
        image_names: list[str] | None = None,
    ) -> list[ImageEvalResult]:
        """
        image_names を指定すると、その画像のみを評価する(train/test分割用)。
        Noneの場合はannotations.json内の全画像を評価する。
        """
        annotations = self._load_annotations()
        results: list[ImageEvalResult] = []

        target_names = (
            image_names if image_names is not None else list(annotations.keys())
        )

        for image_name in target_names:
            record = annotations.get(image_name)
            if record is None:
                logger.warning(f"アノテーションが見つかりません: {image_name}")
                continue

            try:
                result = self.evaluate_one(image_name, record, dp_config=dp_config)
            except Exception:
                logger.exception(f"評価中にエラーが発生しました: {image_name}")
                continue

            if result is not None:
                results.append(result)

        return results

    def summarize(self, results: list[ImageEvalResult]) -> dict:
        total_truth = sum(r.n_truth for r in results)
        total_predicted = sum(r.n_predicted for r in results)
        total_matched = sum(r.n_matched for r in results)
        total_missed = sum(r.n_missed for r in results)
        total_extra = sum(r.n_extra for r in results)

        precision = total_matched / total_predicted if total_predicted else 0.0
        recall = total_matched / total_truth if total_truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        all_errors = [
            abs(t - p)
            for r in results
            for t, p in zip(r.matched_truth_xs, r.matched_pred_xs)
        ]
        mean_abs_error = float(np.mean(all_errors)) if all_errors else None

        return {
            "n_images": len(results),
            "total_truth": total_truth,
            "total_predicted": total_predicted,
            "total_matched": total_matched,
            "total_missed": total_missed,
            "total_extra": total_extra,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_abs_error": mean_abs_error,
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    evaluator = Evaluator()
    results = evaluator.evaluate_all()

    for r in results:
        err_str = (
            f"{r.mean_abs_error:6.1f}px" if r.mean_abs_error is not None else "   N/A "
        )
        print(
            f"{r.image_name:20s} truth={r.n_truth:2d} pred={r.n_predicted:2d} "
            f"matched={r.n_matched:2d} missed={r.n_missed:2d} extra={r.n_extra:2d} "
            f"err={err_str}"
        )

    summary = evaluator.summarize(results)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
