"""
select suitable border line from candidates

候補境界線(Borderline = list[[x, y], ...])の中から、
「単体コスト」と「隣接する境界線間の文字幅の妥当性」を
同時に考慮し、DPで最適な部分集合(採用する境界線の列)を選ぶ。
"""

import logging
from dataclasses import dataclass

from experiments.CharSeg.protetypes.context import Context, Borderline


@dataclass
class DPselecterConfig:
    # 期待する文字幅の区間 [min_width, max_width]（px単位）。
    # 区間内であればペナルティ0、外れるとその逸脱量に応じてペナルティが増える。
    min_char_width: int = 40
    max_char_width: int = 80

    # 区間外に出たときのペナルティの重み。
    # penalty = width_penalty_weight * (区間外への逸脱量)
    width_penalty_weight: float = 1.0

    allow_empty_selection: bool = False


class DPselecter:
    def __init__(self, config: DPselecterConfig | None = None) -> None:
        self.cfg = config or DPselecterConfig()

    def process(self, context: Context) -> None:
        logging.info("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates = context.candidates

        if candidates is None:
            raise ValueError("contextのcandidatesがNoneです。")

        context.selected = self.select_borderline(
            candidates,
            costs=context.candidate_costs,
        )

    def select_borderline(
        self, candidates: list[Borderline], *, costs: list[float] | None = None
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
        #    dp[i]      : 候補iを「最後に採用した」と仮定した場合の最小累積コスト
        #    parent[i]  : dp[i]を実現する直前に採用した候補のインデックス（無ければNone）
        NEG_INF = float("inf")
        dp = [NEG_INF] * n
        parent: list[int | None] = [None] * n

        for i in range(n):
            # 「iより前に何も採用していない」場合（iが最初の採用候補）
            best = sorted_costs[i]
            best_parent = None

            # 「jを直前に採用していた」場合（jはiより前の任意の候補）
            for j in range(i):
                if dp[j] == NEG_INF:
                    continue
                penalty = self._width_penalty(sorted_xs[j], sorted_xs[i])
                cand_cost = dp[j] + penalty + sorted_costs[i]
                if cand_cost < best:
                    best = cand_cost
                    best_parent = j

            dp[i] = best
            parent[i] = best_parent

        # 4. 終端（最後に採用する候補）を決める。
        #    「何も採用しない」を許容する場合は、その選択肢（コスト0）も比較対象に入れる。
        end_idx: int | None = None
        end_cost = 0.0 if self.cfg.allow_empty_selection else NEG_INF

        for i in range(n):
            if dp[i] < end_cost:
                end_cost = dp[i]
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

        logging.info(
            "DP選択結果: %d候補中 %d本を採用 (総コスト=%.3f)",
            n,
            len(selected_sorted_indices),
            dp[end_idx],
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

    def _width_penalty(self, x_prev: float, x_curr: float) -> float:
        """
        隣接する2本の境界線間の文字幅が、期待区間
        [min_char_width, max_char_width] からどれだけ逸脱しているかに
        応じたペナルティを返す。区間内なら0。
        """
        width = x_curr - x_prev

        if width < self.cfg.min_char_width:
            deviation = self.cfg.min_char_width - width
        elif width > self.cfg.max_char_width:
            deviation = width - self.cfg.max_char_width
        else:
            return 0.0

        return self.cfg.width_penalty_weight * deviation
