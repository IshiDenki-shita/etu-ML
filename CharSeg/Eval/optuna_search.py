"""
DPselectorConfigのパラメータをOptunaでベイズ最適化する。
一晩など長時間動かし続け、F1スコアを最大化するパラメータを探す。

フルグリッドサーチと違い、評価済みの試行結果から「次に試すべき
良さそうな値」を推測しながら探索するため、連続値を非常に細かく
探索しても組み合わせ爆発しない。

中断・再開:
    --storage で指定したSQLiteファイルに全試行が保存されるため、
    Ctrl+Cで中断しても、同じ--study-nameで再実行すれば続きから探索できる。

実行方法:
    python -m CharSeg.Eval.optuna_search --hours 8
    (Ctrl+Cでいつでも中断可。その時点のベストを表示して終了する)

必要パッケージ:
    pip install optuna
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import optuna

from CharSeg.DPselector import DPselectorConfig
from CharSeg.Eval.evaluate import Evaluator

logger = logging.getLogger(__name__)


def make_objective(evaluator: Evaluator):
    def objective(trial: optuna.Trial) -> float:
        width_bonus = trial.suggest_float("width_bonus", 300.0, 2500.0)
        min_cw_ratio = trial.suggest_float("min_cw_ratio", 0.3, 0.9)
        max_cw_ratio = trial.suggest_float("max_cw_ratio", min_cw_ratio + 0.05, 2.5)
        weight_scale = trial.suggest_float("weight_scale", 0.2, 2.5)
        narrow_penalty_divisor = trial.suggest_float(
            "narrow_penalty_divisor", 3.0, 100.0
        )
        wide_penalty_divisor = trial.suggest_float("wide_penalty_divisor", 3.0, 100.0)

        dp_config = DPselectorConfig(
            width_bonus=width_bonus,
            min_cw_ratio=min_cw_ratio,
            max_cw_ratio=max_cw_ratio,
            weight_scale=weight_scale,
            narrow_penalty_divisor=narrow_penalty_divisor,
            wide_penalty_divisor=wide_penalty_divisor,
        )

        results = evaluator.evaluate_all(dp_config=dp_config)
        summary = evaluator.summarize(results)

        trial.set_user_attr("precision", summary["precision"])
        trial.set_user_attr("recall", summary["recall"])
        trial.set_user_attr("mean_abs_error", summary["mean_abs_error"] or 0.0)

        return summary["f1"]

    return objective


def make_progress_logger(log_every: int):
    """
    一定試行数おきに進捗をログ出力するコールバック。
    nohupでバックグラウンド実行し、tail -f でログを追う際の確認用。
    """
    start_time = time.time()

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number % log_every != 0:
            return
        elapsed_min = (time.time() - start_time) / 60
        best = study.best_value if study.trials else None
        if best is not None:
            print(
                f"[{elapsed_min:7.1f}分経過] 試行{trial.number:6d}件完了  "
                f"現在のベストF1={best:.4f}",
                flush=True,
            )
        else:
            print(f"[{elapsed_min:7.1f}分経過] 試行{trial.number:6d}件完了", flush=True)

    return callback


def print_best(study: optuna.Study, top_n: int = 10) -> None:
    trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )

    print(f"\n総試行数: {len(study.trials)} (有効な値を持つもの: {len(trials)})")

    if not trials:
        print("有効な試行がありません。")
        return

    print(f"\n=== Top {min(top_n, len(trials))} ===")
    header = (
        f"{'F1':>6} {'P':>6} {'R':>6} {'err':>7} {'width_bonus':>11} "
        f"{'min_r':>6} {'max_r':>6} {'w_scale':>7} {'narrow_div':>10} {'wide_div':>8}"
    )
    print(header)
    print("-" * len(header))

    for t in trials[:top_n]:
        p = t.user_attrs.get("precision", float("nan"))
        r = t.user_attrs.get("recall", float("nan"))
        err = t.user_attrs.get("mean_abs_error", 0.0)
        params = t.params
        print(
            f"{t.value:>6.3f} {p:>6.3f} {r:>6.3f} {err:>7.2f} "
            f"{params['width_bonus']:>11.1f} {params['min_cw_ratio']:>6.3f} "
            f"{params['max_cw_ratio']:>6.3f} {params['weight_scale']:>7.3f} "
            f"{params['narrow_penalty_divisor']:>10.2f} {params['wide_penalty_divisor']:>8.2f}"
        )

    best = trials[0]
    print("\n=== Best Params (DPselectorConfigにそのまま貼り付け可能) ===")
    print("DPselectorConfig(")
    for k, v in best.params.items():
        print(f"    {k}={v!r},")
    print(")")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPselectorのパラメータをOptunaで探索する"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=8.0,
        help="探索を実行する時間(時間単位、デフォルト8時間)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///CharSeg/Eval/optuna_study.db",
        help="中断・再開のための永続化先(SQLite)",
    )
    parser.add_argument("--study-name", type=str, default="dpselector_tuning")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="この試行数おきに進捗をログ出力する(バックグラウンド実行時の確認用)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    evaluator = Evaluator()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(evaluator)
    timeout_seconds = args.hours * 3600

    # ファイルやパイプにリダイレクトしている場合(tty以外)は、
    # \rで行を上書きするプログレスバーは無意味な大量ログになるため無効化する。
    use_progress_bar = sys.stdout.isatty()
    progress_logger = make_progress_logger(args.log_every)

    print(
        f"{args.hours}時間({timeout_seconds:.0f}秒)探索します。"
        f"Ctrl+Cで中断しても結果は {args.storage} に保存されています。"
        f"同じ --study-name で再実行すれば続きから探索できます。\n",
        flush=True,
    )

    try:
        study.optimize(
            objective,
            timeout=timeout_seconds,
            show_progress_bar=use_progress_bar,
            callbacks=[progress_logger],
        )
    except KeyboardInterrupt:
        print("\n中断されました。これまでの最良値を表示します。", flush=True)

    print_best(study)


if __name__ == "__main__":
    main()
