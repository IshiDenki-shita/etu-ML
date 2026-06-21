"""
A* による境界線候補列挙。文字上端の窪みから経路探索がスタート
経路探索を行う幅を制限

パイプライン: app.py → AstarHollow
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments.CharSeg.protetypes.GenCandidates.ConnectLine import (
    Borderline,
    borderline_to_points,
)
from experiments.CharSeg.protetypes.context import Context

logger = logging.getLogger(__name__)


@dataclass
class AstarConfig:
    # cost-map making
    char_pixel_cost: float = 50
    # Astar stating point
    Astar_start_density: float = 1 / 30
    # Astar running
    search_window_ratio: float = 1 / 8


MOVES: tuple[tuple[int, int, float], ...] = (
    # (x, y, step_cost)
    (-1, 1, 1.414),
    (0, 1, 1.0),
    (1, 1, 1.414),
    (-1, 0, 5.0),
    (1, 0, 5.0),
)


class AstarInterval:
    """ステップ4: blank_trimmed から list[Borderline] 候補を列挙する。"""

    def __init__(
        self,
        config: AstarConfig | None = None,
        debug: bool = False,
    ) -> None:

        self.cfg = config or AstarConfig()
        self.debug = debug

    def process(self, context: Context) -> None:
        logging.info(
            "A*アルゴリズムを用いて文字の分割境界線の候補を列挙します。(谷点から開始)"
        )

        blank_trimmed = context.blank_trimmed

        if blank_trimmed is None:
            raise ValueError("context の blank_trimmed が None です")

        window_width = int(blank_trimmed.shape[0] // self.cfg.search_window_ratio)

        start_points = self.raise_Astarting_points(blank_trimmed)

        _, cost_map = self.build_cost_maps(blank_trimmed)
        borderlines, costs = self.generate_candidates(
            cost_map, start_points, window_width
        )

        context.candidates = borderlines
        context.candidate_costs = costs
        logging.info("%d 本の境界線候補を列挙しました", len(borderlines))

        if self.debug:
            self.visualize_candidates(
                blank_trimmed,
                borderlines,
                costs=costs,
            )

    def build_cost_maps(self, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        binary_bool = binary > 0
        dist = dist = np.asarray(distance_transform_edt(~binary_bool), dtype=np.float64)

        if dist is None:
            raise ValueError("distがNoneです。")  # このraiseがないと下の行で警告

        cost = 10.0 / (dist + 1e-3)
        cost = cost.astype(np.float64, copy=False)
        cost[binary_bool] = self.cfg.char_pixel_cost
        return dist.astype(np.float64), cost

    def raise_Astarting_points(self, binary: np.ndarray):
        start_points = []
        w, h = binary.shape
        Astar_interval = int(w * self.cfg.Astar_start_density)
        for i in range(0, w, Astar_interval):
            start_points.append((0, i))
        return start_points

    def astar_best_path(
        self, cost_map: np.ndarray, start_point: tuple[int, int], window_width: int
    ):
        """
        start from a point and then end at bottom
        """
        h, w = cost_map.shape

        # A* fundamental
        best_score_map = np.full((h, w), np.inf, dtype=np.float64)
        visited = np.zeros((h, w), dtype=bool)

        # around generating result
        best_goal_cost = np.inf
        best_goal_point = (0, 0)
        parent_x = np.full((h, w), -1, dtype=np.int32)
        parent_y = np.full((h, w), -1, dtype=np.int32)

        # starting point
        sx, sy = start_point
        c0 = cost_map[sy, sx]  # cost of starting point
        f0 = c0 + float(h - 1 - 0)  # huristic value (ゴールまでの距離の目安・概算)
        heap = [(f0, c0, sx, sy)]  # heap[i] = (huristic_val, whole_cost, x, y)
        best_score_map[sy, sx] = c0

        # limit searching width
        right_wall = sx + window_width // 2
        left_wall = sx - window_width // 2
        right_wall = right_wall if right_wall <= w else w
        left_wall = left_wall if left_wall >= 0 else 0

        while heap:
            f, c, x, y = heapq.heappop(heap)

            if visited[y, x]:
                continue
            if c > best_score_map[y, x]:
                continue

            visited[y, x] = True

            if y == h - 1:  # if reached to the goal
                if c < best_goal_cost:
                    best_goal_cost = c
                    best_goal_point = (x, y)
                if best_goal_point is not None and f >= best_goal_cost:
                    # there is no better path than this
                    break
                continue

            for stepx, stepy, step_cost in MOVES:
                nx = x + stepx
                ny = y + stepy

                if nx < left_wall or nx >= right_wall or ny < 0 or ny >= h:
                    continue
                if visited[ny, nx]:
                    continue

                nc = c + step_cost + cost_map[ny, nx]
                if nc >= best_score_map[ny, nx]:
                    continue

                best_score_map[ny, nx] = nc
                nf = nc + float(h - 1 - ny)
                parent_x[ny, nx] = x
                parent_y[ny, nx] = y
                heapq.heappush(heap, (nf, nc, nx, ny))

        gx, gy = best_goal_point
        best_path = self._reconstruct_path(parent_x, parent_y, gx, gy)
        return best_path, best_goal_cost

    def _reconstruct_path(
        self,
        parent_x: np.ndarray,
        parent_y: np.ndarray,
        gx: int,
        gy: int,
    ) -> np.ndarray:
        xs: list[int] = []
        ys: list[int] = []
        x, y = gx, gy
        while True:
            xs.append(x)
            ys.append(y)
            px = parent_x[y, x]
            py = parent_y[y, x]
            if px < 0:
                break
            x, y = int(px), int(py)
        xs.reverse()
        ys.reverse()
        return np.stack([xs, ys], axis=1).astype(np.int32)

    def generate_candidates(
        self, cost_map: np.ndarray, start_points: list, window_width: int
    ) -> tuple[list[Borderline], list[float]]:

        borderlines: list[Borderline] = []
        costs: list[float] = []

        for start_point in tqdm(start_points, disable=not self.debug):

            points, cost = self.astar_best_path(
                cost_map,
                start_point,
                window_width,
            )

            borderlines.append(points.tolist())
            costs.append(cost)

        return borderlines, costs

    def visualize_candidates(
        self,
        image: np.ndarray,
        borderlines: list[Borderline],
        costs: list[float] | None = None,
    ):
        fig, ax = plt.subplots(figsize=(8, 10))

        ax.imshow(image, cmap="gray")

        if costs is None:
            costs = [0.0] * len(borderlines)

        norm = Normalize(min(costs), max(costs) + 1e-8)
        cmap = plt.get_cmap("viridis")

        for path, cost in zip(borderlines, costs):
            pts = np.asarray(path)
            if len(pts) == 0:
                continue

            color = cmap(norm(cost))

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )

            ax.scatter(
                pts[0, 0],
                pts[0, 1],
                c=[color],
                s=12,
                marker="o",
            )

            ax.scatter(
                pts[-1, 0],
                pts[-1, 1],
                c=[color],
                s=18,
                marker="x",
            )

        ax.set_title("A* candidate paths")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
