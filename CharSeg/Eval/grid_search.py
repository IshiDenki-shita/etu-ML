"""
DPselectorConfigの全パラメータをグリッドサーチし、
evaluate.py の評価指標(precision/recall/f1/mean_abs_error)を比較する。

過学習対策として、annotations.json内の画像をtrain/testに分割し、
trainでグリッドサーチ、testで最終確認する2段階評価を行う。
(54枚というデータ数を考慮し、train:test = 40:14 程度を想定)

探索軸:
    width_bonus             : ボーナスの満額値
    min_cw_ratio             : 文字幅下限 (画像高さ h に対する比率)
    max_cw_ratio             : 文字幅上限 (同上)
    weight_scale             : ペナルティ重みのスケール
    narrow_penalty_divisor   : 狭すぎる方向のペナルティ緩和係数
    wide_penalty_divisor     : 広すぎる方向のペナルティ緩和係数 (asymmetricのみ使用)
    gaussian_sigma_ratio     : gaussian形状のσ (max-min に対する比率)
    bonus_shape              : linear(対称) / gaussian / rect / asymmetric(非対称)
    rep_x_method              : center / mean / median

A*候補生成はDPselectorのパラメータに依存しないため、
Evaluatorのcontextキャッシュにより画像ごとに1回しか実行されない。

実行方法:
    python -m CharSeg.Eval.grid_search
    python -m CharSeg.Eval.grid_search --seed 42 --test-ratio 0.25
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
from dataclasses import dataclass, field

from CharSeg.DPselector import DPselectorConfig
from CharSeg.Eval.evaluate import Evaluator, EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class GridSearchSpace:
    # グリッドサーチで試す値の一覧。各リストが1軸。全軸の直積を探索する。
    width_bonus: list[float] = field(default_factory=lambda: [700.0, 900.0, 1200.0])
    min_cw_ratio: list[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75])
    max_cw_ratio: list[float] = field(default_factory=lambda: [1.10, 1.30, 1.60, 2.00])
    weight_scale: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    narrow_penalty_divisor: list[float] = field(
        default_factory=lambda: [10.0, 20.0, 50.0]
    )
    # asymmetric の時のみ使用 (linearはnarrow_penalty_divisorのみ使う対称版)
    wide_penalty_divisor: list[float] = field(
        default_factory=lambda: [10.0, 20.0, 50.0]
    )
    # gaussian の時のみ使用。狭い値=鋭いピーク、広い値=なだらか
    gaussian_sigma_ratio: list[float] = field(
        default_factory=lambda: [0.15, 0.3, 0.5, 0.8, 1.2]
    )
    bonus_shape: list[str] = field(
        default_factory=lambda: ["linear", "gaussian", "rect", "asymmetric"]
    )
    rep_x_method: list[str] = field(
        default_factory=lambda: ["center", "mean", "median"]
    )


def split_train_test(
    image_names: list[str], test_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(image_names)
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_ratio))
    test_names = shuffled[:n_test]
    train_names = shuffled[n_test:]
    return train_names, test_names


def iter_valid_combos(space: GridSearchSpace):
    """
    軸の直積から、不正な組み合わせ(min_cw_ratio >= max_cw_ratio)と
    無関係パラメータの重複組み合わせ(例: linearでwide_penalty_divisorを
    振っても結果は同じになる)を除いた、評価する価値のある組み合わせだけを返す。
    """
    seen_keys: set[tuple] = set()

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
    ) in itertools.product(
        space.width_bonus,
        space.min_cw_ratio,
        space.max_cw_ratio,
        space.weight_scale,
        space.narrow_penalty_divisor,
        space.wide_penalty_divisor,
        space.gaussian_sigma_ratio,
        space.bonus_shape,
        space.rep_x_method,
    ):
        if min_ratio >= max_ratio:
            continue

        # bonus_shapeごとに「実際に効くパラメータ」だけをキーに含めることで、
        # 無関係パラメータの値違いによる重複評価を除外する。
        if bonus_shape == "linear":
            # wide_penalty_divisorは使われない、gaussian_sigma_ratioも使われない
            key = (
                "linear",
                width_bonus,
                min_ratio,
                max_ratio,
                weight_scale,
                narrow_div,
                bonus_shape,
                rep_x_method,
            )
        elif bonus_shape == "asymmetric":
            # gaussian_sigma_ratioは使われない
            key = (
                "asymmetric",
                width_bonus,
                min_ratio,
                max_ratio,
                weight_scale,
                narrow_div,
                wide_div,
                bonus_shape,
                rep_x_method,
            )
        elif bonus_shape == "rect":
            # narrow/wide_divisor, gaussian_sigma_ratio, weight_scaleは使われない
            key = ("rect", width_bonus, min_ratio, max_ratio, rep_x_method)
        elif bonus_shape == "gaussian":
            # narrow/wide_divisor, weight_scaleは使われない
            key = (
                "gaussian",
                width_bonus,
                min_ratio,
                max_ratio,
                gaussian_sigma,
                rep_x_method,
            )
        else:
            key = (
                bonus_shape,
                width_bonus,
                min_ratio,
                max_ratio,
                weight_scale,
                narrow_div,
                wide_div,
                gaussian_sigma,
                rep_x_method,
            )

        if key in seen_keys:
            continue
        seen_keys.add(key)

        yield (
            width_bonus,
            min_ratio,
            max_ratio,
            weight_scale,
            narrow_div,
            wide_div,
            gaussian_sigma,
            bonus_shape,
            rep_x_method,
        )


def run_grid_search(
    eval_config: EvalConfig | None = None,
    space: GridSearchSpace | None = None,
    image_names: list[str] | None = None,
) -> list[dict]:
    """
    image_names を指定すると、その画像群のみで評価する(train側に使う)。
    Noneなら全画像を使う。
    """
    space = space or GridSearchSpace()
    evaluator = Evaluator(eval_config)

    results: list[dict] = []
    combos = list(iter_valid_combos(space))
    n_total = len(combos)
    print(f"探索する組み合わせ数(重複除去後): {n_total}")

    for done, (
        width_bonus,
        min_ratio,
        max_ratio,
        weight_scale,
        narrow_div,
        wide_div,
        gaussian_sigma,
        bonus_shape,
        rep_x_method,
    ) in enumerate(combos, start=1):

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

        eval_results = evaluator.evaluate_all(
            dp_config=dp_config, image_names=image_names
        )
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

        if done % 200 == 0 or done == n_total:
            best_f1 = max(r["f1"] for r in results)
            print(f"  {done}/{n_total} 完了  現在のベストF1={best_f1:.4f}")

    results.sort(key=lambda r: (r["f1"], r["recall"]), reverse=True)
    return results


def print_results(results: list[dict], top_n: int = 30) -> None:
    header = (
        f"{'F1':>6} {'P':>6} {'R':>6} {'err':>6} "
        f"{'shape':>11} {'rep_x':>6} "
        f"{'bonus':>6} {'min_r':>5} {'max_r':>5} "
        f"{'w_sc':>5} {'n_div':>6} {'w_div':>6} {'g_sig':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in results[:top_n]:
        err = r["mean_abs_error"] if r["mean_abs_error"] is not None else 0.0
        print(
            f"{r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {err:>6.1f} "
            f"{r['bonus_shape']:>11} {r['rep_x_method']:>6} "
            f"{r['width_bonus']:>6.0f} {r['min_cw_ratio']:>5.2f} {r['max_cw_ratio']:>5.2f} "
            f"{r['weight_scale']:>5.2f} {r['narrow_div']:>6.1f} {r['wide_div']:>6.1f} "
            f"{r['gaussian_sigma']:>5.2f}"
        )


def print_grouped_best(results: list[dict]) -> None:
    print("\n=== bonus_shape 別ベスト(train) ===")
    for shape in ["linear", "gaussian", "rect", "asymmetric"]:
        shape_results = [r for r in results if r["bonus_shape"] == shape]
        if shape_results:
            best = shape_results[0]
            err = best["mean_abs_error"] or 0.0
            print(
                f"{shape:>11}: F1={best['f1']:.3f}  P={best['precision']:.3f}  "
                f"R={best['recall']:.3f}  err={err:.1f}px"
            )

    print("\n=== rep_x_method 別ベスト(train) ===")
    for method in ["center", "mean", "median"]:
        method_results = [r for r in results if r["rep_x_method"] == method]
        if method_results:
            best = method_results[0]
            err = best["mean_abs_error"] or 0.0
            print(
                f"{method:>6}: F1={best['f1']:.3f}  P={best['precision']:.3f}  "
                f"R={best['recall']:.3f}  err={err:.1f}px"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPselectorのパラメータをグリッドサーチする"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.25,
        help="test用に取り分ける画像の割合(デフォルト0.25 ≒ 54枚なら14枚程度)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/test分割の乱数シード(再現性のため固定推奨)",
    )
    parser.add_argument(
        "--top-n-for-test",
        type=int,
        default=5,
        help="trainでのF1上位何件をtestで再評価するか",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    evaluator_for_split = Evaluator()
    all_names = evaluator_for_split.all_image_names()
    train_names, test_names = split_train_test(all_names, args.test_ratio, args.seed)

    print(
        f"全{len(all_names)}枚 -> train {len(train_names)}枚 / test {len(test_names)}枚"
    )
    print(f"test画像: {test_names}\n")

    # --- train: グリッドサーチ ---
    print("=== Train: グリッドサーチ実行 ===")
    train_results = run_grid_search(image_names=train_names)
    print(f"\nF1上位(train)を表示します。\n")
    print_results(train_results, top_n=30)
    print_grouped_best(train_results)

    # --- test: train上位の組み合わせのみ再評価 ---
    print(f"\n=== Test: trainでのF1上位{args.top_n_for_test}件をtestデータで再評価 ===")
    evaluator_for_test = Evaluator()
    test_eval_rows: list[dict] = []

    for r in train_results[: args.top_n_for_test]:
        dp_config = DPselectorConfig(
            width_bonus=r["width_bonus"],
            min_cw_ratio=r["min_cw_ratio"],
            max_cw_ratio=r["max_cw_ratio"],
            weight_scale=r["weight_scale"],
            narrow_penalty_divisor=r["narrow_div"],
            wide_penalty_divisor=r["wide_div"],
            gaussian_sigma_ratio=r["gaussian_sigma"],
            bonus_shape=r["bonus_shape"],  # type: ignore[arg-type]
            rep_x_method=r["rep_x_method"],  # type: ignore[arg-type]
        )
        test_results = evaluator_for_test.evaluate_all(
            dp_config=dp_config, image_names=test_names
        )
        test_summary = evaluator_for_test.summarize(test_results)
        test_eval_rows.append(
            {
                "train_f1": r["f1"],
                "test_f1": test_summary["f1"],
                "test_precision": test_summary["precision"],
                "test_recall": test_summary["recall"],
                "test_err": test_summary["mean_abs_error"] or 0.0,
                "bonus_shape": r["bonus_shape"],
                "rep_x_method": r["rep_x_method"],
                "width_bonus": r["width_bonus"],
                "min_cw_ratio": r["min_cw_ratio"],
                "max_cw_ratio": r["max_cw_ratio"],
            }
        )

    print(
        f"{'train_F1':>9} {'test_F1':>8} {'test_P':>7} {'test_R':>7} {'test_err':>9} "
        f"{'shape':>11} {'rep_x':>6} {'bonus':>6} {'min_r':>5} {'max_r':>5}"
    )
    for row in test_eval_rows:
        print(
            f"{row['train_f1']:>9.3f} {row['test_f1']:>8.3f} {row['test_precision']:>7.3f} "
            f"{row['test_recall']:>7.3f} {row['test_err']:>9.1f} "
            f"{row['bonus_shape']:>11} {row['rep_x_method']:>6} "
            f"{row['width_bonus']:>6.0f} {row['min_cw_ratio']:>5.2f} {row['max_cw_ratio']:>5.2f}"
        )

    print(
        "\n(train_F1とtest_F1の差が大きい設定は、trainデータへの過学習を"
        "疑った方が良いです。差が小さく、test_F1自体も高い設定を採用してください。)"
    )


if __name__ == "__main__":
    main()
