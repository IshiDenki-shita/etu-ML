import logging
from typing import Optional, Literal
from dataclasses import dataclass, field
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from CharSeg.context import Context, Borderline

logger = logging.getLogger(__name__)

# ボーナス形状の選択肢
BonusShape = Literal["linear", "gaussian", "rect", "asymmetric"]

# 代表x座標の計算方法の選択肢
RepXMethod = Literal["center", "mean", "median"]


@dataclass
class DPselectorConfig:
    width_bonus: float = 700.0
    width_penalty_weight: Optional[float] = None  # 文字幅の予想値より決定
    allow_empty_selection: bool = False

    # 文字幅の予想値 (画像高さ h に対する比率)
    min_cw_ratio: float = 0.8
    max_cw_ratio: float = 1.0
    # width_penalty_weight = weight_scale * (width_bonus / min_char_width)
    weight_scale: float = 1.2

    # _width_bonus の形状
    #   linear     : 二次減衰 + 0クリップ (従来の挙動)
    #   gaussian   : ガウス型減衰 (中心付近を強く優遇、0クリップなし)
    #   rect       : 矩形 (区間内なら満額、外は即0)
    #   asymmetric : 狭すぎる方向は急減衰、広すぎる方向はなだらか
    bonus_shape: BonusShape = "linear"

    # linear / asymmetric で使う: deviation = (ズレ)^2 / divisor
    narrow_penalty_divisor: float = 20.0
    wide_penalty_divisor: float = 10.0

    # gaussian で使う: σ = (max_cw - min_cw) * gaussian_sigma_ratio
    gaussian_sigma_ratio: float = 0.5

    # _representative_x の計算方法
    #   center : y中央に最も近い点のx (従来の挙動)
    #   mean   : 全点のx座標の平均
    #   median : 全点のx座標の中央値 (外れ値耐性あり)
    rep_x_method: RepXMethod = "center"


class DPselector:
    def __init__(self, debug: bool, cfg: DPselectorConfig | None = None) -> None:
        self.debug = debug
        self.cfg = cfg if cfg is not None else DPselectorConfig()
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
            self.cfg.width_bonus / max(1, min_char_width)
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
                best = -sorted_costs[i]
                best_parent = None
            else:
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

        end_idx: int | None = None
        end_score = NEG_INF

        for i in range(n):
            if score[i] > end_score:
                end_score = score[i]
                end_idx = i

        if end_idx is None:
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

    # -------------------------
    # 代表x座標
    # -------------------------
    def _representative_x(self, borderline: Borderline) -> float:
        if not borderline:
            raise ValueError("空のBorderlineが渡されました。")

        method = self.cfg.rep_x_method

        if method == "mean":
            return float(np.mean([p[0] for p in borderline]))

        if method == "median":
            return float(np.median([p[0] for p in borderline]))

        # "center": y中央に最も近い点のx (デフォルト・従来の挙動)
        ys = [p[1] for p in borderline]
        y_center = (min(ys) + max(ys)) / 2
        closest_point = min(borderline, key=lambda p: abs(p[1] - y_center))
        return float(closest_point[0])

    # -------------------------
    # 幅ボーナス
    # -------------------------
    def _width_bonus(
        self, x_prev: float, x_curr: float, expected_cw: tuple[int, int]
    ) -> float:
        width = x_curr - x_prev
        max_cw, min_cw = expected_cw
        shape = self.cfg.bonus_shape

        if shape == "rect":
            # 区間内なら満額、外は即0
            return self.cfg.width_bonus if min_cw <= width <= max_cw else 0.0

        if shape == "gaussian":
            # ガウス型: 区間中心からの距離に応じてなだらかに減衰（0クリップなし）
            center = (min_cw + max_cw) / 2.0
            sigma = (max_cw - min_cw) * self.cfg.gaussian_sigma_ratio
            if sigma <= 0:
                sigma = 1.0
            return float(
                self.cfg.width_bonus * np.exp(-((width - center) ** 2) / (2 * sigma**2))
            )

        if shape == "asymmetric":
            # 狭すぎる方向: 急減衰 (narrow_penalty_divisorで調整)
            # 広すぎる方向: なだらか (wide_penalty_divisorで調整)
            if width < min_cw:
                deviation = (min_cw - width) ** 2 / self.cfg.narrow_penalty_divisor
            elif width > max_cw:
                deviation = (width - max_cw) ** 2 / self.cfg.wide_penalty_divisor
            else:
                return self.cfg.width_bonus
            assert self.cfg.width_penalty_weight
            return max(
                0.0, self.cfg.width_bonus - self.cfg.width_penalty_weight * deviation
            )

        # "linear": 二次減衰 + 0クリップ (デフォルト・従来の挙動)
        if width < min_cw:
            deviation = (min_cw - width) ** 2 / self.cfg.narrow_penalty_divisor
        elif width > max_cw:
            deviation = (width - max_cw) ** 2 / self.cfg.wide_penalty_divisor
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

        fig.suptitle(
            f"DPselector Visualization  [shape={self.cfg.bonus_shape}, rep_x={self.cfg.rep_x_method}]",
            fontsize=11,
        )
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
        ax.set_title(f"2. Representative X (method={self.cfg.rep_x_method})")
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
        ax.set_title(f"3. width_bonus (shape={self.cfg.bonus_shape})")

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
