"""
select suitable border line from candidates

候補境界線(Borderline = list[[x, y], ...])の中から、
「単体コスト」と「隣接する境界線間の文字幅の妥当性」を
同時に考慮し、DPで最適な部分集合(採用する境界線の列)を選ぶ。
"""

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np

from CharSeg.context import Context, Borderline

logger = logging.getLogger(__name__)


@dataclass
class DPselecterConfig:
    # 隣接ペアの幅が期待区間に収まっているときに与えるボーナス（スコアに加算）。
    # このボーナスが単体コストより大きいほど、「繋いだ方が得」になりやすい。
    width_bonus: float = 400.0

    expected_width_ratio = 1

    # 区間外に出たときの減衰の重み。
    # ボーナス = max(0, width_bonus - width_penalty_weight * 逸脱量)
    width_penalty_weight: float = 5.0

    allow_empty_selection: bool = False


class DPselecter:
    # 期待する文字幅の区間 [min_width, max_width]（px単位）。
    # 区間内であれば「繋ぐボーナス」が満額、外れるとその逸脱量に応じて減衰する。
    min_char_width: int
    max_char_width: int

    def __init__(self, config: DPselecterConfig | None = None) -> None:
        self.cfg = config or DPselecterConfig()

    def process(self, context: Context) -> None:
        logger.debug("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates = context.candidates
        blank_trimmed = context.blank_trimmed

        if candidates is None or blank_trimmed is None:
            raise ValueError("contextのcandidatesがNoneです。")

        self.adopt_char_width(blank_trimmed)

        context.selected = self.select_borderline(
            candidates,
            costs=context.candidate_costs,
        )

    def adopt_char_width(self, blank_trimmed: np.ndarray):
        h, w = blank_trimmed.shape

        self.min_char_width = int(h * 1.2)
        self.max_char_width = int(h * 0.8)
        logger.debug(
            f"DPselecter: char width {self.min_char_width} ~ {self.max_char_width}"
        )

    def select_borderline(
        self,
        candidates: list[Borderline],
        *,
        costs: list[float] | None = None,
    ) -> list[Borderline]:

        if not candidates:
            return []

        if costs is None or len(costs) != len(candidates):
            logging.warning(
                "candidate_costsが不正のため、単体コストを0として扱います。"
            )
            costs = [0.0] * len(candidates)

        # 1. 各候補の代表x座標（y中央に最も近い点のx座標）を求める。
        rep_xs = [self._representative_x(b) for b in candidates]

        # 2. 代表x座標の昇順に候補をソートする（DPは左から右へ処理するため）。
        order = sorted(range(len(candidates)), key=lambda i: rep_xs[i])
        sorted_costs = [costs[i] for i in order]
        sorted_xs = [rep_xs[i] for i in order]
        n = len(order)

        # 3. DP本体。
        #    score[i]   : 候補iを「最後に採用した」と仮定した場合の最良スコア（大きいほど良い）
        #                 = -(単体コストの合計) + (隣接ペアの繋ぐボーナスの合計)
        #    parent[i]  : score[i]を実現する直前に採用した候補のインデックス（無ければNone）
        NEG_INF = float("-inf")
        score = [NEG_INF] * n
        parent: list[int | None] = [None] * n

        for i in range(n):
            # 「iより前に何も採用していない」場合（iが最初の採用候補）
            # スコアは単体コストのみが差し引かれる（繋ぐボーナスはまだ無い）。
            best = -sorted_costs[i]
            best_parent = None

            # 「jを直前に採用していた」場合（jはiより前の任意の候補）
            for j in range(i):
                if score[j] == NEG_INF:
                    continue
                bonus = self._width_bonus(sorted_xs[j], sorted_xs[i])
                cand_score = score[j] + bonus - sorted_costs[i]
                if cand_score > best:
                    best = cand_score
                    best_parent = j

            score[i] = best
            parent[i] = best_parent

        # 4. 終端（最後に採用する候補）を決める。
        #    「何も採用しない」を許容する場合は、その選択肢（スコア0）も比較対象に入れる。
        end_idx: int | None = None
        end_score = 0.0 if self.cfg.allow_empty_selection else NEG_INF

        for i in range(n):
            if score[i] > end_score:
                end_score = score[i]
                end_idx = i

        if end_idx is None:
            # 候補が1つもない、または「何も採用しない」が最良だった場合
            return []

        # 5. バックトラックして採用インデックス列（昇順）を復元する。
        selected_sorted_indices: list[int] = []
        cur: int | None = end_idx
        while cur is not None:
            selected_sorted_indices.append(cur)
            cur = parent[cur]
        selected_sorted_indices.reverse()

        logger.debug(
            f"{parent}\nDP選択結果: {n}候補中 {len(selected_sorted_indices)}本を採用 (総スコア={score[end_idx]})",
        )

        # ソート後インデックス -> 元のcandidatesのインデックスへ変換して返す
        return [candidates[order[i]] for i in selected_sorted_indices]

    def _representative_x(self, borderline: Borderline) -> float:
        """
        境界線([x, y]の点列)から、y中央に最も近い点のx座標を代表値として返す。
        """
        if not borderline:
            raise ValueError("空のBorderlineが渡されました。")

        ys = [p[1] for p in borderline]
        y_center = (min(ys) + max(ys)) / 2

        closest_point = min(borderline, key=lambda p: abs(p[1] - y_center))
        return float(closest_point[0])

    def _width_bonus(self, x_prev: float, x_curr: float) -> float:
        """
        隣接する2本の境界線間の文字幅が、期待区間
        [min_char_width, max_char_width] にどれだけ合致しているかに
        応じたボーナス（スコアへの加点）を返す。

        区間内なら満額(width_bonus)、外れるほど線形に減衰し、
        0未満には下がらない(繋ぐことが「損」にしかならない事態を避けるため)。
        """
        width = x_curr - x_prev

        if width < self.min_char_width:
            deviation = self.min_char_width - width
        elif width > self.max_char_width:
            deviation = width - self.max_char_width
        else:
            return self.cfg.width_bonus

        bonus = self.cfg.width_bonus - self.cfg.width_penalty_weight * deviation
        return max(0.0, bonus)
