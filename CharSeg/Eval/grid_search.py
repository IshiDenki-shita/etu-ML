"""
DPselectorConfigの (width_bonus, min_cw_ratio, max_cw_ratio) をグリッドサーチし、
evaluate.py の評価指標(precision/recall/f1/mean_abs_error)を比較する。

A*候補生成(Preprocesser~AstarInterval)はパラメータに依存しないため、
Evaluatorのcontextキャッシュにより画像ごとに1回しか実行されない。
組み合わせ数が多い場合でも、繰り返し部分はDP選択のみで済む。

実行方法:
    python -m CharSeg.Eval.grid_search
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from CharSeg.DPselector import DPselectorConfig
from CharSeg.Eval.evaluate import Evaluator, EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class GridSearchSpace:
    width_bonus: list[float] = field(
        default_factory=lambda: [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0]
    )
    min_cw_ratio: list[float] = field(
        default_factory=lambda: [0.55, 0.60, 0.65, 0.70, 0.75]
    )
    max_cw_ratio: list[float] = field(default_factory=lambda: [1.3])
    weight_scale: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.4, 1.6])


def run_grid_search(
    eval_config: EvalConfig | None = None,
    space: GridSearchSpace | None = None,
) -> list[dict]:
    space = space or GridSearchSpace()
    evaluator = Evaluator(eval_config)

    results: list[dict] = []

    combos = list(
        itertools.product(
            space.width_bonus,
            space.min_cw_ratio,
            space.max_cw_ratio,
            space.weight_scale,
        )
    )

    for width_bonus, min_ratio, max_ratio, weight_scale in combos:
        if min_ratio >= max_ratio:
            continue  # 不正な組み合わせはスキップ

        dp_config = DPselectorConfig(
            width_bonus=width_bonus,
            min_cw_ratio=min_ratio,
            max_cw_ratio=max_ratio,
            weight_scale=weight_scale,
        )

        eval_results = evaluator.evaluate_all(dp_config=dp_config)
        summary = evaluator.summarize(eval_results)

        results.append(
            {
                "width_bonus": width_bonus,
                "min_cw_ratio": min_ratio,
                "max_cw_ratio": max_ratio,
                "weight_scale": weight_scale,
                **summary,
            }
        )

    results.sort(key=lambda r: (r["f1"], r["recall"]), reverse=True)
    return results


def print_results(results: list[dict], top_n: int = 20) -> None:
    header = (
        f"{'width_bonus':>11} {'min_ratio':>9} {'max_ratio':>9} {'w_scale':>7} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'err(px)':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results[:top_n]:
        err = r["mean_abs_error"] if r["mean_abs_error"] is not None else 0.0
        print(
            f"{r['width_bonus']:>11.0f} {r['min_cw_ratio']:>9.2f} {r['max_cw_ratio']:>9.2f} "
            f"{r['weight_scale']:>7.2f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} {err:>7.2f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # 出力を見やすくするためログを抑制

    results = run_grid_search()
    print(f"{len(results)} 件の組み合わせを評価しました。F1上位を表示します。\n")
    print_results(results)


if __name__ == "__main__":
    main()
