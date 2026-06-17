"""
A* による境界線候補列挙。

パイプライン: app.py → AstarCandidateGenerator
単体検証:

    python -m experiments.CharSeg.protetypes.GenCandidates.Astar
    python -m experiments.CharSeg.protetypes.GenCandidates.Astar --image path/to/cell.png
"""

from __future__ import annotations

import argparse
import heapq
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments.CharSeg.protetypes.GenCandidates.ConnectLine import (
    Borderline,
    borderline_to_points,
)
from experiments.CharSeg.protetypes.context import Context

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


@dataclass
class AstarContext:
    original: np.ndarray | None = None
    binary: np.ndarray | None = None
    distance_map: np.ndarray | None = None
    cost_map: np.ndarray | None = None
    candidates: list[Borderline] | None = None
    candidate_costs: list[float] | None = None


def build_cost_maps(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """binary: 0=background, 255=foreground."""
    binary_bool = binary > 0
    dist = distance_transform_edt(~binary_bool)
    cost = 1.0 / (dist + 1e-3)
    cost = cost.astype(np.float64, copy=False)
    cost[binary_bool] = 1e6
    return dist.astype(np.float64), cost


def _heuristic(y: int, goal_y: int) -> float:
    return float(goal_y - y)


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
        f0 = g0 + _heuristic(0, goal_y)
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
            nf = tentative + _heuristic(ny, goal_y)
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
        color=255,
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
) -> tuple[list[Borderline], list[float]]:
    penalty_map = np.zeros_like(cost_map, dtype=np.float64)
    candidates: list[Borderline] = []
    costs: list[float] = []

    for _ in tqdm(range(num_candidates)):
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
class AstarConfig:
    num_candidates: int = NUM_CANDIDATES
    suppression_radius: int = SUPPRESSION_RADIUS
    suppression_penalty: float = SUPPRESSION_PENALTY


class AstarCandidateGenerator:
    """ステップ3: line_removed から list[Borderline] 候補を列挙する。"""

    def __init__(
        self,
        config: AstarConfig | None = None,
        debug: bool = False,
    ) -> None:
        self.config = config or AstarConfig()
        self.debug = debug

    def process(self, context: Context) -> None:
        logging.info("A*アルゴリズムを用いて文字の分割境界線の候補を列挙します。")

        blank_trimmed = context.blank_trimmed

        if blank_trimmed is None:
            raise ValueError("context の blank_trimmed が None です")

        _, cost_map = build_cost_maps(blank_trimmed)
        borderlines, costs = generate_candidates(
            cost_map,
            num_candidates=self.config.num_candidates,
            suppression_radius=self.config.suppression_radius,
            suppression_penalty=self.config.suppression_penalty,
        )

        context.candidates = borderlines
        context.candidate_costs = costs
        logging.info("%d 本の境界線候補を列挙しました", len(borderlines))

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
    cmap = plt.cm.tab10

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


def _load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return img


def _to_binary(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def make_synthetic_binary(height: int = 100, width: int = 72) -> np.ndarray:
    """検証用: 中央に谷がある二筋の文字セル風画像。"""
    img = np.zeros((height, width), dtype=np.uint8)
    img[:, 14:20] = 255
    img[:, 52:58] = 255
    return img


def run_experiment(
    binary: np.ndarray,
    *,
    original: np.ndarray | None = None,
    num_candidates: int = NUM_CANDIDATES,
    show_plot: bool = True,
) -> AstarContext:
    context = AstarContext()
    context.original = original
    context.binary = binary

    dist, cost = build_cost_maps(binary)
    context.distance_map = dist
    context.cost_map = cost

    borderlines, costs = generate_candidates(
        cost,
        num_candidates=num_candidates,
    )
    context.candidates = borderlines
    context.candidate_costs = costs

    print(len(context.candidates))
    for i, path_cost in enumerate(costs):
        print(f"candidate[{i}].cost = {path_cost:.6f}")

    if show_plot:
        visualize_candidates(
            binary,
            context.candidates,
            costs=costs,
            original=original,
        )

    return context


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="A* suppression-based Top-K borderline candidate enumeration",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="入力グレースケール/カラー画像（未指定時は合成バイナリ）",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=NUM_CANDIDATES,
        help=f"列挙する候補数（既定: {NUM_CANDIDATES}）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="matplotlib 表示をスキップ",
    )
    args = parser.parse_args()

    if args.image is not None:
        original = _load_grayscale(args.image)
        if original.ndim == 2:
            binary = _to_binary(original)
        else:
            raise ValueError("2次元グレースケール画像を想定しています")
        run_experiment(
            binary,
            original=original,
            num_candidates=args.num_candidates,
            show_plot=not args.no_show,
        )
    else:
        binary = make_synthetic_binary()
        run_experiment(
            binary,
            original=binary,
            num_candidates=args.num_candidates,
            show_plot=not args.no_show,
        )


if __name__ == "__main__":
    main()
