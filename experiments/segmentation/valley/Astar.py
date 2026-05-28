import heapq
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

# =========================================================
# Context
# =========================================================


@dataclass
class AstarContext:
    original: np.ndarray | None = None
    binary: np.ndarray | None = None
    distance_map: np.ndarray | None = None
    cost_map: np.ndarray | None = None
    path: np.ndarray | None = None


# =========================================================
# Fast A*
# =========================================================


class FastAstarSegmenter:
    def process(self, context: AstarContext) -> AstarContext:
        if context.binary is None:
            raise ValueError("binary is required")

        context.distance_map = self._distance_map(context.binary)

        context.cost_map = self._cost_map(
            binary=context.binary,
            distance_map=context.distance_map,
        )

        H, W = context.binary.shape

        start_y = 0
        start_x = W // 2

        goal_y = H - 1
        goal_x = W // 2

        context.path = self._astar(
            context.cost_map,
            start_y,
            start_x,
            goal_y,
            goal_x,
        )

        return context

    # =====================================================
    # Distance Map
    # =====================================================

    def _distance_map(self, binary: np.ndarray) -> np.ndarray:
        binary_bool = binary > 0

        dist = distance_transform_edt(~binary_bool)

        return dist.astype(np.float32)

    # =====================================================
    # Cost Map
    # =====================================================

    def _cost_map(
        self,
        binary: np.ndarray,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        binary_bool = binary > 0

        cost = 1.0 / (distance_map + 1e-3)

        cost[binary_bool] = 1e6

        return cost.astype(np.float32)

    # =====================================================
    # Heuristic (Octile Distance)
    # =====================================================

    def _heuristic(
        self,
        y,
        x,
        gy,
        gx,
    ):
        dx = abs(gx - x)
        dy = abs(gy - y)

        return max(dx, dy) + (1.41421356 - 1.0) * min(dx, dy)

    # =====================================================
    # Optimized A*
    # =====================================================

    def _astar(
        self,
        cost_map: np.ndarray,
        sy: int,
        sx: int,
        gy: int,
        gx: int,
    ):
        H, W = cost_map.shape

        INF = np.float32(1e30)

        g_score = np.full((H, W), INF, dtype=np.float32)

        visited = np.zeros((H, W), dtype=np.uint8)

        parent_y = np.full((H, W), -1, dtype=np.int32)
        parent_x = np.full((H, W), -1, dtype=np.int32)

        g_score[sy, sx] = 0.0

        pq = []

        heapq.heappush(
            pq,
            (
                0.0,
                sy,
                sx,
            ),
        )

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.41421356),
            (-1, 1, 1.41421356),
            (1, -1, 1.41421356),
            (1, 1, 1.41421356),
        ]

        cost_local = cost_map
        g_local = g_score
        visited_local = visited

        while pq:
            _, cy, cx = heapq.heappop(pq)

            # ============================================
            # Skip duplicated node
            # ============================================

            if visited_local[cy, cx]:
                continue

            visited_local[cy, cx] = 1

            # ============================================
            # Goal
            # ============================================

            if cy == gy and cx == gx:
                break

            current_g = g_local[cy, cx]

            # ============================================
            # Expand
            # ============================================

            for dy, dx, move_cost in neighbors:
                ny = cy + dy
                nx = cx + dx

                if ny < 0 or ny >= H:
                    continue

                if nx < 0 or nx >= W:
                    continue

                if visited_local[ny, nx]:
                    continue

                tentative = current_g + cost_local[ny, nx] * move_cost

                if tentative < g_local[ny, nx]:
                    g_local[ny, nx] = tentative

                    parent_y[ny, nx] = cy
                    parent_x[ny, nx] = cx

                    h = self._heuristic(
                        ny,
                        nx,
                        gy,
                        gx,
                    )

                    f = tentative + h

                    heapq.heappush(
                        pq,
                        (
                            f,
                            ny,
                            nx,
                        ),
                    )

        # =================================================
        # Reconstruct
        # =================================================

        path = []

        cy = gy
        cx = gx

        while not (cy == sy and cx == sx):
            path.append((cy, cx))

            py = parent_y[cy, cx]
            px = parent_x[cy, cx]

            if py == -1:
                break

            cy = py
            cx = px

        path.append((sy, sx))

        path.reverse()

        return np.array(path)


# =========================================================
# Main
# =========================================================

img = cv2.imread(
    "photos/sample/cells/chikuten.jpeg",
    cv2.IMREAD_GRAYSCALE,
)

if img is None:
    raise ValueError("画像を取得できませんでした。")

_, binary = cv2.threshold(
    img,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

context = AstarContext(
    original=img,
    binary=binary,
)

segmenter = FastAstarSegmenter()

context = segmenter.process(context)

# =========================================================
# Visualization
# =========================================================

vis = cv2.cvtColor(
    context.original,
    cv2.COLOR_GRAY2BGR,
)

for y, x in context.path:
    vis[y, x] = (0, 0, 255)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(8, 8),
)

axes[0].imshow(context.original, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(context.binary, cmap="gray")
axes[1].set_title("Binary")

axes[2].imshow(context.distance_map, cmap="jet")
axes[2].set_title("Distance Transform")

axes[3].imshow(vis[..., ::-1])
axes[3].set_title("Fast A*")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
