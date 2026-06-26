"""
A* による境界線候補列挙。

パイプライン: app.py → AstarNormal
"""

from __future__ import annotations

import heapq
import logging
from typing import cast
from dataclasses import dataclass

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from CharSeg.GenCandidates.ConnectLine import (
    Borderline,
    borderline_to_points,
)
from CharSeg.context import Context

logger = logging.getLogger(__name__)

# --- parameters ---
NUM_CANDIDATES = 20
SUPPRESSION_RADIUS = 10
SUPPRESSION_PENALTY = 500.0

# (dx, dy, step_cost) — 上方向は禁止
MOVES: tuple[tuple[int, int, float], ...] = (
    (-1, 1, 1.414),
    (0, 1, 1.0),
    (1, 1, 1.414),
    (-1, 0, 5.0),
    (1, 0, 5.0),
)


def build_cost_maps(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary_bool = binary > 0
    dist = distance_transform_edt(~binary_bool)
    dist = cast(np.ndarray, distance_transform_edt(~binary_bool))
    cost = 10.0 / (dist + 1e-3)
    cost = cost.astype(np.float64, copy=False)
    cost[binary_bool] = 1e6
    return dist.astype(np.float64), cost


def astar_best_path(
    cost_map: np.ndarray,
    penalty_map: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """
    y=0 の全列から出発し、y=H-1 のいずれかへ到達する最良経路を返す。
    戻り値: (points (N,2) int32, total_cost) または (None, None)
    """
    h, w = cost_map.shape
    goal_y = h - 1
    effective = cost_map + penalty_map

    g_score = np.full((h, w), np.inf, dtype=np.float64)
    visited = np.zeros((h, w), dtype=bool)
    parent_x = np.full((h, w), -1, dtype=np.int32)
    parent_y = np.full((h, w), -1, dtype=np.int32)

    heap: list[tuple[float, float, int, int]] = []

    for x in range(w):
        g0 = effective[0, x]
        g_score[0, x] = g0
        f0 = g0 + float(goal_y - 0)
        heapq.heappush(heap, (f0, g0, x, 0))

    best_goal: tuple[int, int] | None = None
    best_g = np.inf

    while heap:
        f, g, x, y = heapq.heappop(heap)

        if visited[y, x]:
            continue
        if g > g_score[y, x]:
            continue

        visited[y, x] = True

        if y == goal_y:
            if g < best_g:
                best_g = g
                best_goal = (x, y)
            if best_goal is not None and f >= best_g:
                break
            continue

        for dx, dy, step_cost in MOVES:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if visited[ny, nx]:
                continue

            tentative = g + step_cost + effective[ny, nx]
            if tentative >= g_score[ny, nx]:
                continue

            g_score[ny, nx] = tentative
            parent_x[ny, nx] = x
            parent_y[ny, nx] = y
            nf = tentative + float(goal_y - ny)
            heapq.heappush(heap, (nf, tentative, nx, ny))

    if best_goal is None:
        return None, None

    gx, gy = best_goal
    points = _reconstruct_path(parent_x, parent_y, gx, gy)
    return points, float(best_g)


def _reconstruct_path(
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


def points_to_borderline(points: np.ndarray) -> Borderline:
    """(N, 2) int32 配列 → [[x, y], ...]"""
    return points.tolist()


def apply_suppression(
    points: np.ndarray,
    penalty_map: np.ndarray,
    *,
    radius: int = SUPPRESSION_RADIUS,
    penalty: float = SUPPRESSION_PENALTY,
) -> None:
    mask = np.zeros(penalty_map.shape, dtype=np.uint8)
    poly = points.reshape(-1, 1, 2)
    thickness = max(1, radius * 2 + 1)
    cv2.polylines(
        mask,
        [poly],
        isClosed=False,
        color=[255],
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    penalty_map[mask > 0] += penalty


def generate_candidates(
    cost_map: np.ndarray,
    *,
    num_candidates: int = NUM_CANDIDATES,
    suppression_radius: int = SUPPRESSION_RADIUS,
    suppression_penalty: float = SUPPRESSION_PENALTY,
    debug: bool,
) -> tuple[list[Borderline], list[float]]:

    penalty_map = np.zeros_like(cost_map, dtype=np.float64)
    candidates: list[Borderline] = []
    costs: list[float] = []

    for _ in tqdm(range(num_candidates), disable=not debug):
        points, cost = astar_best_path(cost_map, penalty_map)
        if points is None or cost is None:
            logger.warning("これ以上経路が見つかりません（%d 本）", len(candidates))
            break

        candidates.append(points_to_borderline(points))
        costs.append(cost)
        apply_suppression(
            points,
            penalty_map,
            radius=suppression_radius,
            penalty=suppression_penalty,
        )

    return candidates, costs


@dataclass
class AstarNormalConfig:
    num_candidates: int = NUM_CANDIDATES
    suppression_radius: int = SUPPRESSION_RADIUS
    suppression_penalty: float = SUPPRESSION_PENALTY


class AstarNormal:
    """ステップ3: line_removed から list[Borderline] 候補を列挙する。"""

    def __init__(
        self,
        config: AstarNormalConfig | None = None,
        debug: bool = False,
    ) -> None:
        self.config = config or AstarNormalConfig()
        self.debug = debug

    def process(self, context: Context) -> None:
        logger.debug("A*アルゴリズムを用いて文字の分割境界線の候補を列挙します。")

        blank_trimmed = context.blank_trimmed

        if blank_trimmed is None:
            raise ValueError("context の blank_trimmed が None です")

        _, cost_map = build_cost_maps(blank_trimmed)
        borderlines, costs = generate_candidates(
            cost_map,
            num_candidates=self.config.num_candidates,
            suppression_radius=self.config.suppression_radius,
            suppression_penalty=self.config.suppression_penalty,
            debug=self.debug,
        )

        context.candidates = borderlines
        context.candidate_costs = costs
        logger.debug("%d 本の境界線候補を列挙しました", len(borderlines))

        if self.debug:
            visualize_candidates(
                blank_trimmed,
                borderlines,
                costs=costs,
            )


def visualize_candidates(
    binary: np.ndarray,
    candidates: list[Borderline],
    *,
    costs: list[float] | None = None,
    original: np.ndarray | None = None,
) -> None:
    base = original if original is not None else binary
    if base.ndim == 3:
        display = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    else:
        display = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(display)
    cmap = plt.get_cmap("tab10")

    for i, cand in enumerate(candidates):
        color = cmap(i % 10)
        pts = borderline_to_points(cand)
        label = f"candidate[{i}]"
        if costs is not None and i < len(costs):
            label += f" cost={costs[i]:.1f}"
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_title(f"A* borderline candidates (K={len(candidates)})")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
