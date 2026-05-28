import heapq
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt

# =========================================================
# A*
# =========================================================


def astar(cost_map, start, goal):
    H, W = cost_map.shape

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    pq = []

    heapq.heappush(pq, (0, start))

    came_from = {}

    g_score = {start: 0}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            break

        cy, cx = current

        for dy, dx in neighbors:
            ny = cy + dy
            nx = cx + dx

            if not (0 <= ny < H and 0 <= nx < W):
                continue

            move_cost = cost_map[ny, nx]

            if dy != 0 and dx != 0:
                move_cost *= 1.414

            tentative = g_score[current] + move_cost

            neighbor = (ny, nx)

            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative

                h = abs(goal[0] - ny) + abs(goal[1] - nx)

                f = tentative + h

                heapq.heappush(pq, (f, neighbor))

                came_from[neighbor] = current

    # reconstruct
    path = []

    current = goal

    while current in came_from:
        path.append(current)
        current = came_from[current]

    path.append(start)

    path.reverse()

    return path


img = cv2.imread("photos/sample/cells/chikuten.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("画像を取得できませんでした。")

_, binary = cv2.threshold(
    img,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

binary_bool = binary > 0

dist = distance_transform_edt(~binary_bool)
dist = dist.astype(np.float32)

# 谷中央を低コストにする
cost = 1.0 / (dist + 1e-3)

# 文字内部は超高コスト
cost[binary_bool] = 1e6

H, W = binary.shape

start = (H // 2, 0)
goal = (H // 2, W - 1)

path = astar(cost, start, goal)


vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for y, x in path:
    vis[y, x] = (0, 0, 255)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Binary")

axes[2].imshow(dist, cmap="jet")
axes[2].set_title("Distance Transform")

axes[3].imshow(vis[..., ::-1])
axes[3].set_title("A* Path")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
