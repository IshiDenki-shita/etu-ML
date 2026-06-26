# valley pointsなど点群を境界線として捉えるユーティリティ

from typing import List, Tuple
from collections import deque
import cv2
import numpy as np

from CharSeg.context import Borderline


def borderline_to_points(borderline: Borderline) -> np.ndarray:
    return np.asarray(borderline, dtype=np.int32).reshape(-1, 2)


def rasterize_borderlines(
    shape: tuple[int, int],
    borderlines: list[Borderline],
    *,
    thickness: int = 1,
) -> np.ndarray:
    """折れ線群を (H, W) のマスク画像にラスタライズする。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for borderline in borderlines:
        if len(borderline) < 2:
            continue
        poly = borderline_to_points(borderline).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [poly],
            isClosed=False,
            color=[255],
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
    return mask


def points_as_line(valley_point_map: np.ndarray) -> List[List]:
    h, w = valley_point_map.shape
    visited = np.zeros((h, w), dtype=bool)

    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    lines = []
    for i in range(h):
        for j in range(w):

            if visited[i, j]:
                continue
            if not valley_point_map[i, j]:
                continue

            queue = deque([(i, j)])
            visited[i, j] = True
            line = []

            while queue:
                y, x = queue.popleft()
                line.append((y, x))

                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx

                    if ny < 0 or ny >= h:
                        continue
                    if nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx]:
                        continue
                    if not valley_point_map[ny, nx]:
                        continue

                    visited[ny, nx] = True
                    queue.append((ny, nx))

            lines.append(line)

    return lines


def line_from_points(self, lines: List[List]) -> List[List]:
    verticals = []

    for line in lines:
        ...

    return verticals
