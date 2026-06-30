"""
DPselectorConfigの全パラメータをグリッドサーチし、
evaluate.py の評価指標(precision/recall/f1/mean_abs_error)を比較する。

探索軸:
    width_bonus         : ボーナスの満額値
    min_cw_ratio        : 文字幅下限 (画像高さ h に対する比率)
    max_cw_ratio        : 文字幅上限 (同上)
    weight_scale        : ペナルティ重みのスケール
    narrow_penalty_divisor : 狭すぎる方向のペナルティ緩和係数
    wide_penalty_divisor   : 広すぎる方向のペナルティ緩和係数
    gaussian_sigma_ratio   : gaussian形状のσ (max-min に対する比率)
    bonus_shape         : linear / gaussian / rect / asymmetric
    rep_x_method        : center / mean / median

A*候補生成はDPselectorのパラメータに依存しないため、
Evaluatorのcontextキャッシュにより画像ごとに1回しか実行されない。

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
    # グリッドサーチで試す値の一覧。
    # 各リストが1軸。全軸の直積を探索する。
    width_bonus: list[float] = field(default_factory=lambda: [700.0, 900.0, 1200.0])
    min_cw_ratio: list[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75])
    max_cw_ratio: list[float] = field(default_factory=lambda: [1.30])
    weight_scale: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    narrow_penalty_divisor: list[float] = field(
        default_factory=lambda: [10.0, 20.0, 50.0]
    )
    wide_penalty_divisor: list[float] = field(
        default_factory=lambda: [10.0, 20.0, 50.0]
    )
    gaussian_sigma_ratio: list[float] = field(
        default_factory=lambda: [0.5]  # gaussian以外では使われないが軸として存在
    )
    bonus_shape: list[str] = field(
        default_factory=lambda: ["linear", "gaussian", "rect", "asymmetric"]
    )
    rep_x_method: list[str] = field(
        default_factory=lambda: ["center", "mean", "median"]
    )


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
            space.narrow_penalty_divisor,
            space.wide_penalty_divisor,
            space.gaussian_sigma_ratio,
            space.bonus_shape,
            space.rep_x_method,
        )
    )

    n_total = sum(
        1 for c in combos if c[1] < c[2]  # min_cw_ratio < max_cw_ratio のもののみ
    )
    print(f"探索する組み合わせ数: {n_total}")

    done = 0
    for (
        width_bonus,
        min_ratio,
        max_ratio,
        weight_scale,
        narrow_div,
        wide_div,
        gaussian_sigma,
        bonus_shape,
        rep_x_method,
    ) in combos:
        if min_ratio >= max_ratio:
            continue

        dp_config = DPselectorConfig(
            width_bonus=width_bonus,
            min_cw_ratio=min_ratio,
            max_cw_ratio=max_ratio,
            weight_scale=weight_scale,
            narrow_penalty_divisor=narrow_div,
            wide_penalty_divisor=wide_div,
            gaussian_sigma_ratio=gaussian_sigma,
            bonus_shape=bonus_shape,  # type: ignore[arg-type]
            rep_x_method=rep_x_method,  # type: ignore[arg-type]
        )

        eval_results = evaluator.evaluate_all(dp_config=dp_config)
        summary = evaluator.summarize(eval_results)

        results.append(
            {
                "width_bonus": width_bonus,
                "min_cw_ratio": min_ratio,
                "max_cw_ratio": max_ratio,
                "weight_scale": weight_scale,
                "narrow_div": narrow_div,
                "wide_div": wide_div,
                "gaussian_sigma": gaussian_sigma,
                "bonus_shape": bonus_shape,
                "rep_x_method": rep_x_method,
                **summary,
            }
        )

        done += 1
        if done % 100 == 0:
            best_f1 = max(r["f1"] for r in results)
            print(f"  {done}/{n_total} 完了  現在のベストF1={best_f1:.4f}")

    results.sort(key=lambda r: (r["f1"], r["recall"]), reverse=True)
    return results


def print_results(results: list[dict], top_n: int = 30) -> None:
    header = (
        f"{'F1':>6} {'P':>6} {'R':>6} {'err':>6} "
        f"{'shape':>11} {'rep_x':>6} "
        f"{'bonus':>6} {'min_r':>5} {'max_r':>5} "
        f"{'w_sc':>5} {'n_div':>6} {'w_div':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results[:top_n]:
        err = r["mean_abs_error"] if r["mean_abs_error"] is not None else 0.0
        print(
            f"{r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {err:>6.1f} "
            f"{r['bonus_shape']:>11} {r['rep_x_method']:>6} "
            f"{r['width_bonus']:>6.0f} {r['min_cw_ratio']:>5.2f} {r['max_cw_ratio']:>5.2f} "
            f"{r['weight_scale']:>5.2f} {r['narrow_div']:>6.1f} {r['wide_div']:>6.1f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    results = run_grid_search()
    print(f"\nF1上位を表示します。\n")
    print_results(results, top_n=30)

    # bonus_shape ごとのベストを表示
    print("\n=== bonus_shape 別ベスト ===")
    for shape in ["linear", "gaussian", "rect", "asymmetric"]:
        shape_results = [r for r in results if r["bonus_shape"] == shape]
        if shape_results:
            best = shape_results[0]
            err = best["mean_abs_error"] or 0.0
            print(
                f"{shape:>11}: F1={best['f1']:.3f}  P={best['precision']:.3f}  "
                f"R={best['recall']:.3f}  err={err:.1f}px"
            )

    # rep_x_method ごとのベストを表示
    print("\n=== rep_x_method 別ベスト ===")
    for method in ["center", "mean", "median"]:
        method_results = [r for r in results if r["rep_x_method"] == method]
        if method_results:
            best = method_results[0]
            err = best["mean_abs_error"] or 0.0
            print(
                f"{method:>6}: F1={best['f1']:.3f}  P={best['precision']:.3f}  "
                f"R={best['recall']:.3f}  err={err:.1f}px"
            )


if __name__ == "__main__":
    main()


"""
結果1

@dataclass
class DPselectorConfig:
    width_bonus: float = 1028.146721097928
    width_penalty_weight: Optional[float] = None  # 文字幅の予想値より決定

    # 文字幅の予想値 (画像高さ h に対する比率)
    min_cw_ratio: float = 0.6275017296806547
    max_cw_ratio: float = 1.2130461173653475
    # width_penalty_weight = weight_scale * (width_bonus / min_char_width)
    weight_scale: float = 1.6487968724217137

    # deviation = (理想とのズレ)^2 / divisor
    # 狭すぎる場合(narrow)と広すぎる場合(wide)で別々のスケールを持つ
    narrow_penalty_divisor: float = 8.850340203708514
    wide_penalty_divisor: float = 92.28635357024723
"""
