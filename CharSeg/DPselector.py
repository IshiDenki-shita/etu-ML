import logging
from typing import Optional
from dataclasses import dataclass
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from CharSeg.context import Context, Borderline

logger = logging.getLogger(__name__)


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


class DPselector:
    cfg = DPselectorConfig()

    def __init__(self, debug: bool) -> None:
        self.debug = debug
        self._last_selected_orig_indices: list[int] = []

        if os.name == "nt":
            plt.rcParams["font.family"] = "MS Gothic"
        else:
            plt.rcParams["font.family"] = "Hiragino Sans"

    def process(self, context: Context) -> None:
        logger.debug("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates = context.candidates
        blank_trimmed = context.blank_trimmed

        if candidates is None or blank_trimmed is None:
            raise ValueError("contextのcandidatesがNoneです。")

        max_cw, min_cw = self.adopt_char_width(blank_trimmed)

        candidates_with_left_edge, costs_with_left_edge = self._add_left_edge_candidate(
            candidates, context.candidate_costs, blank_trimmed
        )

        context.selected = self.select_borderline(
            candidates_with_left_edge,
            costs=costs_with_left_edge,
            expected_cw=(max_cw, min_cw),
        )

        if self.debug:
            self.visualize(
                context,
                expected_cw=(max_cw, min_cw),
                candidates=candidates_with_left_edge,
                costs=costs_with_left_edge,
                blank_trimmed=blank_trimmed,
            )

    def _add_left_edge_candidate(
        self,
        candidates: list[Borderline],
        costs: list[float] | None,
        blank_trimmed: np.ndarray,
    ) -> tuple[list[Borderline], list[float]]:
        """
        画像左端(x=0)を通る垂直な境界線を候補の先頭に追加する。
        この候補はDPの中でwidth_bonusの計算対象になるが、
        必ず最終的な選択結果に含まれるよう、後段のDPで
        「採用しない」ルートを作らない前提で扱う。
        """
        h, _ = blank_trimmed.shape
        left_edge_line: Borderline = [[0, 0], [0, h - 1]]

        new_candidates = [left_edge_line] + list(candidates)

        if costs is None or len(costs) != len(candidates):
            new_costs = [0.0] * (len(candidates) + 1)
        else:
            new_costs = [0.0] + list(costs)

        return new_candidates, new_costs

    def adopt_char_width(self, blank_trimmed: np.ndarray):
        h, w = blank_trimmed.shape
        max_char_width = int(h * self.cfg.max_cw_ratio)
        min_char_width = int(h * self.cfg.min_cw_ratio)

        # 満額スコア < max(狭間隔ペナルティ) を満たす
        self.cfg.width_penalty_weight = self.cfg.weight_scale * (
            self.cfg.width_bonus / min_char_width
        )

        logger.debug(f"expected max/min char width {min_char_width} ~ {max_char_width}")
        return max_char_width, min_char_width

    def select_borderline(
        self,
        candidates: list[Borderline],
        *,
        costs: list[float] | None = None,
        expected_cw: tuple[int, int],
    ) -> list[Borderline]:

        if not candidates:
            self._last_selected_orig_indices = []
            return []

        if costs is None or len(costs) != len(candidates):
            logger.warning("candidate_costsが不正のため、単体コストを0として扱います。")
            costs = [0.0] * len(candidates)

        rep_xs = [self._representative_x(b) for b in candidates]

        order = sorted(range(len(candidates)), key=lambda i: rep_xs[i])
        sorted_costs = [costs[i] for i in order]
        sorted_xs = [rep_xs[i] for i in order]
        n = len(order)

        left_edge_sorted_idx = int(np.argmin(sorted_xs))

        NEG_INF = float("-inf")
        score = [NEG_INF] * n
        parent: list[int | None] = [None] * n

        for i in range(n):
            if i == left_edge_sorted_idx:
                # left_edgeは「何も採用していない状態」から必ず採用される
                # 唯一の起点。これにより、left_edgeを経由しないスコアの
                # 系列は作られなくなる。
                best = -sorted_costs[i]
                best_parent = None
            else:
                # left_edge以外は「起点」になれない(必ずどこかのjを経由する)。
                # ただしjがleft_edge自身であるケースも通常のループで含まれる。
                best = NEG_INF
                best_parent = None

                for j in range(i):
                    if score[j] == NEG_INF:
                        continue
                    bonus = self._width_bonus(sorted_xs[j], sorted_xs[i], expected_cw)
                    cand_score = score[j] + bonus - sorted_costs[i]
                    if cand_score > best:
                        best = cand_score
                        best_parent = j

            score[i] = best
            parent[i] = best_parent

        # 終端探索: left_edgeを経由していない候補(score=NEG_INF)は
        # 自動的に選択対象から除外される。
        end_idx: int | None = None
        end_score = NEG_INF

        for i in range(n):
            if score[i] > end_score:
                end_score = score[i]
                end_idx = i

        if end_idx is None:
            # left_edge自身しか候補がない、またはすべてNEG_INFの場合
            self._last_selected_orig_indices = [order[left_edge_sorted_idx]]
            return [candidates[order[left_edge_sorted_idx]]]

        selected_sorted_indices: list[int] = []
        cur: int | None = end_idx
        while cur is not None:
            selected_sorted_indices.append(cur)
            cur = parent[cur]
        selected_sorted_indices.reverse()

        self._last_selected_orig_indices = [order[i] for i in selected_sorted_indices]

        return [candidates[order[i]] for i in selected_sorted_indices]

    def _representative_x(self, borderline: Borderline) -> float:
        if not borderline:
            raise ValueError("空のBorderlineが渡されました。")

        ys = [p[1] for p in borderline]
        y_center = (min(ys) + max(ys)) / 2

        closest_point = min(borderline, key=lambda p: abs(p[1] - y_center))
        return float(closest_point[0])

    def _width_bonus(
        self, x_prev: float, x_curr: float, expected_cw: tuple[int, int]
    ) -> float:
        width = x_curr - x_prev
        max_char_width, min_char_width = expected_cw

        if width < min_char_width:
            deviation = (min_char_width - width) ** 2 / self.cfg.narrow_penalty_divisor
        elif width > max_char_width:
            deviation = (width - max_char_width) ** 2 / self.cfg.wide_penalty_divisor
        else:
            return self.cfg.width_bonus

        assert self.cfg.width_penalty_weight
        bonus = self.cfg.width_bonus - self.cfg.width_penalty_weight * deviation
        return max(0.0, bonus)

    # -------------------------
    # Visualization
    # -------------------------
    def visualize(
        self,
        context: Context,
        *,
        expected_cw: tuple[int, int],
        candidates: list[Borderline] | None = None,
        costs: list[float] | None = None,
        blank_trimmed: np.ndarray | None = None,
    ) -> None:
        if blank_trimmed is None:
            blank_trimmed = context.blank_trimmed
        if candidates is None:
            candidates = context.candidates
        if costs is None:
            costs = context.candidate_costs

        if blank_trimmed is None or candidates is None:
            raise ValueError("visualize()に必要なcontextの値がNoneです。")

        if costs is None or len(costs) != len(candidates):
            logger.warning("candidate_costsが無効です。コスト色分けを0で代用します。")
            costs = [0.0] * len(candidates)

        max_char_width, min_char_width = expected_cw
        image_height, image_width = blank_trimmed.shape

        selected_indices = self._last_selected_orig_indices
        if not selected_indices and context.selected:
            selected_indices = self._fallback_match_indices(
                candidates, context.selected
            )

        rep_xs_all = [self._representative_x(b) for b in candidates]
        ordered_rep_xs = [rep_xs_all[i] for i in selected_indices]

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
        fig.subplots_adjust(left=0.14, right=0.95, top=0.93, bottom=0.05, hspace=0.5)

        self._draw_candidates_panel(fig, axes[0], blank_trimmed, candidates, costs)
        self._draw_representative_x_panel(
            axes[1], blank_trimmed, candidates, ordered_rep_xs
        )
        self._draw_width_bonus_panel(
            axes[2], ordered_rep_xs, min_char_width, max_char_width, image_width
        )
        self._draw_selected_panel(
            axes[3], blank_trimmed, candidates, costs, selected_indices
        )

        fig.suptitle("DPselector Visualization", fontsize=12)
        plt.show()

    def _fallback_match_indices(
        self, candidates: list[Borderline], selected: list[Borderline]
    ) -> list[int]:
        indices = []
        for sel in selected:
            for i, cand in enumerate(candidates):
                if cand is sel:
                    indices.append(i)
                    break
        return indices

    def _draw_candidates_panel(self, fig, ax, blank_trimmed, candidates, costs):
        ax.imshow(blank_trimmed, cmap="gray")

        vmin, vmax = min(costs), max(costs) + 1e-8
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("plasma")

        for cand, cost in zip(candidates, costs):
            pts = np.asarray(cand)
            if len(pts) == 0:
                continue
            ax.plot(
                pts[:, 0], pts[:, 1], color=cmap(norm(cost)), linewidth=1.4, alpha=0.9
            )

        ax.set_title("1. Candidates (colored by cost)")
        ax.axis("off")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar_ax = ax.inset_axes([-0.16, 0.0, 0.03, 1.0])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label("cost", fontsize=8)

    def _draw_representative_x_panel(
        self, ax, blank_trimmed, candidates, ordered_rep_xs
    ):
        h, w = blank_trimmed.shape
        ax.imshow(blank_trimmed, cmap="gray")

        for cand in candidates:
            pts = np.asarray(cand)
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color="#7EC8E3", linewidth=0.9, alpha=0.35)

        for rep_x in ordered_rep_xs:
            ax.axvline(x=rep_x, color="#FF8C32", linewidth=1.3, alpha=0.95)

        ax.set_title("2. Representative X (adopted method only)")

        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("x (px)", fontsize=8)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    def _draw_width_bonus_panel(
        self, ax, ordered_rep_xs, min_char_width, max_char_width, image_width
    ):
        if len(ordered_rep_xs) == 0:
            ax.text(
                0.5,
                0.5,
                "採用された境界線がありません",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax.set_title("3. width_bonus across image columns")
            ax.axis("off")
            return

        expected_cw = (max_char_width, min_char_width)

        xs = np.arange(image_width)
        bonuses = np.full(image_width, np.nan, dtype=np.float64)

        first_x = ordered_rep_xs[0]
        for x in xs:
            if x < first_x:
                continue

            x_prev = first_x
            for rep_x in ordered_rep_xs:
                if rep_x <= x:
                    x_prev = rep_x
                else:
                    break

            bonuses[x] = self._width_bonus(x_prev, float(x), expected_cw)

        ax.plot(xs, bonuses, color="#FF8C32", linewidth=1.5)
        ax.set_ylabel("width_bonus", color="#FF8C32")
        ax.tick_params(axis="y", labelcolor="#FF8C32")

        for rep_x in ordered_rep_xs:
            ax.axvline(
                x=rep_x, color="#1FA2D6", linewidth=1.0, alpha=0.7, linestyle="--"
            )

        ax.set_xlim(0, image_width)
        ax.set_xlabel("x (px)", fontsize=8)
        ax.set_title(
            "3. width_bonus across image columns (based on previous adopted line)"
        )

    def _draw_selected_panel(
        self, ax, blank_trimmed, candidates, costs, selected_indices
    ):
        ax.imshow(blank_trimmed, cmap="gray")

        selected_set = set(selected_indices)
        unselected = [i for i in range(len(candidates)) if i not in selected_set]

        if unselected:
            cool_cmap = plt.get_cmap("cool")
            for k, i in enumerate(unselected):
                pts = np.asarray(candidates[i])
                if len(pts) == 0:
                    continue
                color = cool_cmap(k / max(1, len(unselected) - 1))
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.9, alpha=0.4)

        if selected_indices:
            warm_cmap = plt.get_cmap("autumn")
            for k, i in enumerate(selected_indices):
                pts = np.asarray(candidates[i])
                if len(pts) == 0:
                    continue
                color = warm_cmap(k / max(1, len(selected_indices) - 1))
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2)

        ax.set_title("4. Selected (warm) vs Unselected (cool)")
        ax.axis("off")
