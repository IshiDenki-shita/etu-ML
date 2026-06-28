import logging
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from CharSeg.context import Context, Borderline

logger = logging.getLogger(__name__)


@dataclass
class DPselectorConfig:
    width_bonus: float = 400.0
    width_penalty_weight: float = 5.0
    allow_empty_selection: bool = False


class DPselector:
    cfg = DPselectorConfig()

    def __init__(self, debug: bool) -> None:
        self.debug = debug
        self._last_selected_orig_indices: list[int] = []
        # Mac用の日本語フォント（ヒラギノ角ゴ）を設定
        plt.rcParams["font.family"] = "Hiragino Sans"

    def process(self, context: Context) -> None:
        logger.debug("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates = context.candidates
        blank_trimmed = context.blank_trimmed

        if candidates is None or blank_trimmed is None:
            raise ValueError("contextのcandidatesがNoneです。")

        max_cw, min_cw = self.adopt_char_width(blank_trimmed)

        context.selected = self.select_borderline(
            candidates, costs=context.candidate_costs, expected_cw=(max_cw, min_cw)
        )

        if self.debug:
            self.visualize(context, expected_cw=(max_cw, min_cw))

    def adopt_char_width(self, blank_trimmed: np.ndarray):
        h, w = blank_trimmed.shape
        max_char_width = int(h * 1.2)
        min_char_width = int(h * 0.8)
        logger.debug(f"DPselecter: char width {min_char_width} ~ {max_char_width}")
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

        NEG_INF = float("-inf")
        score = [NEG_INF] * n
        parent: list[int | None] = [None] * n

        for i in range(n):
            best = -sorted_costs[i]
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
        end_score = 0.0 if self.cfg.allow_empty_selection else NEG_INF

        for i in range(n):
            if score[i] > end_score:
                end_score = score[i]
                end_idx = i

        if end_idx is None:
            self._last_selected_orig_indices = []
            return []

        selected_sorted_indices: list[int] = []
        cur: int | None = end_idx
        while cur is not None:
            selected_sorted_indices.append(cur)
            cur = parent[cur]
        selected_sorted_indices.reverse()

        # 可視化用に「採用された候補の元インデックス（x昇順）」を保持しておく
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
            deviation = min_char_width - width
        elif width > max_char_width:
            deviation = width - max_char_width
        else:
            return self.cfg.width_bonus

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
    ) -> None:
        blank_trimmed = context.blank_trimmed
        candidates = context.candidates
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
        cmap = plt.get_cmap("plasma")  # 暗紫〜明るい黄色（全体的に明るいトーン）

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

        # 背景の全候補線は明るい水色（寒色寄り、薄く）
        for cand in candidates:
            pts = np.asarray(cand)
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color="#7EC8E3", linewidth=0.9, alpha=0.35)

        # 採用されたrepresentative_xは明るいオレンジの縦線
        for rep_x in ordered_rep_xs:
            ax.axvline(x=rep_x, color="#FF8C32", linewidth=1.3, alpha=0.95)

        ax.set_title("2. Representative X (adopted method only)")

        # y軸は消すが、x軸はpx目盛りとして残す
        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # imshowのy反転を維持
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

        # 各列x(0~image_width)に対して、「直前の採用境界線1本」を基準にwidth_bonusを計算する。
        # ・最初の採用境界線より左（まだ基準点が存在しない区間）は計算しない(NaN)。
        # ・最後の採用境界線より右は、最後の採用境界線を基準点として計算を続ける。
        xs = np.arange(image_width)
        bonuses = np.full(image_width, np.nan, dtype=np.float64)

        first_x = ordered_rep_xs[0]
        for x in xs:
            if x < first_x:
                continue  # 左端: 基準点がまだ無いので計算しない

            # xの直前(x以下で最大)の採用境界線を基準点とする
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

        # 採用された境界線の位置に縦線を重ねる
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

        # 不採用候補 = 寒色（明るい水色〜青のグラデーション）
        if unselected:
            cool_cmap = plt.get_cmap("cool")
            for k, i in enumerate(unselected):
                pts = np.asarray(candidates[i])
                if len(pts) == 0:
                    continue
                color = cool_cmap(k / max(1, len(unselected) - 1))
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.9, alpha=0.4)

        # 採用候補 = 暖色（明るい黄色〜オレンジ〜赤のグラデーション）
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
