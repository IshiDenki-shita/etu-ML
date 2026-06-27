import logging
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm

from CharSeg.context import Context

logger = logging.getLogger(__name__)


@dataclass
class LNRConfig:
    # contour vector
    cont_nighr_len: int = 10
    max_theta_thresh: np.float16 = np.deg2rad(2, dtype=np.float16)

    # picking vectors
    min_length_thresh: np.float16 = np.float16(20)
    target_theta_horizontal: np.float16 = np.deg2rad(0, dtype=np.float16)
    target_theta_vertical: np.float16 = np.deg2rad(90, dtype=np.float16)
    target_theta_tolerance: np.float16 = np.deg2rad(3, dtype=np.float16)

    # line connection
    connect_dist_thresh: float = 50
    min_segment_length_ratio: float = 0.3
    direction_estimation_window: int = 5
    direction_alignment_cos_thresh: float = 0.7

    #
    min_char_domain: int = 100


@dataclass
class ContourVectors:
    """1つの輪郭について、座標列と接線ベクトル列をペアで保持する。"""

    points: List[Tuple[int, int]]
    tangents: List[Tuple[float, float]]


@dataclass
class _Endpoint:
    """直線セグメントの端点情報。

    direction は端点における外向きの方向ベクトル(dy, dx、単位ベクトル)。
    端点付近の複数点から主成分方向として推定する。
    """

    segment_idx: int
    is_head: bool
    point: Tuple[int, int]
    direction: Tuple[float, float]


class LineNoiseRemover:
    cfg = LNRConfig()
    image_shape: Tuple

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, context: Context):
        logger.debug("ホワイトボードのマス目の線を取り除きます。")
        binary = context.preprocessed

        if binary is None:
            raise ValueError("contextのpreprocessedがNoneです。")

        self.image_shape = binary.shape

        binary = self.remove_lines(binary, target_theta=0)
        binary = self.remove_lines(binary, target_theta=np.pi / 2)
        context.line_removed = binary

    def remove_lines(self, binary: np.ndarray, target_theta) -> np.ndarray:
        lines = self.detect_lines(binary, target_theta)
        merged_lines = self.merge_lines(lines)
        result = self.erase_lines(binary, merged_lines)
        return result

    def detect_lines(
        self, binary: np.ndarray, target_theta
    ) -> List[List[Tuple[int, int]]]:
        contours = self.detect_contours(binary)
        contour_vectors = self.arrange_contour_vectors2(contours)
        direct_lines = self.detect_direct_line(contour_vectors)
        needed_lines = self.pick_needed_line(direct_lines, target_theta)

        if self.debug:
            self._visualize_detection(
                binary,
                contours,
                contour_vectors,
                direct_lines,
                needed_lines,
                target_theta,
            )

        return needed_lines

    def merge_lines(
        self, lines: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        connectable, standalone = self._split_by_length(lines)

        endpoints = self._collect_endpoints(connectable)
        raw_pairs = self._search_close_pairs(endpoints)
        pairs = self._filter_pairs(endpoints, raw_pairs)
        adjacency = self._build_adjacency(len(endpoints), pairs)

        connected_lines = self._trace_chain(connectable, endpoints, adjacency)

        if self.debug:
            self._visualize_merge(lines, endpoints, pairs, connected_lines)

        return connected_lines + standalone

    def erase_lines(
        self, binary: np.ndarray, lines: List[List[Tuple[int, int]]]
    ) -> np.ndarray:

        h, w = binary.shape
        line_image = self.draw_straight_line((h, w), lines)
        result = self.remove_noise_line(binary, line_image)
        return result

    # -------------------------
    # Detection
    # -------------------------
    def detect_contours(self, binary: np.ndarray):
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,  # only outside edges
            cv2.CHAIN_APPROX_NONE,
        )
        return contours

    def arrange_contour_vectors2(self, contours) -> List[ContourVectors]:
        """
        輪郭ごとに、座標列(points)と接線ベクトル列(tangents)を
        ペアにしたContourVectorsを作る。
        """
        cont_neighr_len = self.cfg.cont_nighr_len
        contour_vectors: List[ContourVectors] = []

        for contour in contours:
            length = len(contour)
            if length < 2 * cont_neighr_len + 1:
                continue

            points: List[Tuple[int, int]] = []
            tangents: List[Tuple[float, float]] = []

            for i in range(length):
                idx1 = (i - cont_neighr_len) % length
                idx2 = (i + cont_neighr_len) % length
                x1, y1 = contour[idx1][0]
                x2, y2 = contour[idx2][0]
                x, y = contour[i][0]

                points.append((y, x))
                tangents.append((x2 - x1, y2 - y1))

            contour_vectors.append(ContourVectors(points=points, tangents=tangents))

        logger.debug(
            f"arranged tangent vectors about {len(contour_vectors)} valid_contours"
        )
        return contour_vectors

    def detect_direct_line(self, contour_vectors: List[ContourVectors]):
        """
        return direct line as like a cv2.findContour()
        """
        direct_lines = []

        for cv_pair in contour_vectors:
            direct_line: List[Tuple[int, int]] = []

            theta_diffs = self.vector_difference_theta(cv_pair.tangents)
            length = len(theta_diffs)

            for j in range(length):
                if abs(theta_diffs[j]) <= self.cfg.max_theta_thresh:
                    direct_line.append(cv_pair.points[j])
                elif len(direct_line) > 2:
                    direct_lines.append(direct_line)
                    direct_line = []

            if len(direct_line) > 2:
                direct_lines.append(direct_line)

        logger.debug(f"detcted {len(direct_lines)} direct_lines")
        return direct_lines

    def vector_difference_theta(self, tan_vecs: List[Tuple[float, float]]):
        """
        the difference theta1 and theta2 as a vector difference
        """
        theta_diffs = []
        length = len(tan_vecs)
        for i in range(length):
            vx1, vy1 = tan_vecs[i]
            vx2, vy2 = tan_vecs[(i + 1) % length]

            norm1 = np.hypot(vx1, vy1)
            norm2 = np.hypot(vx2, vy2)

            if norm1 == 0 or norm2 == 0:
                theta_diffs.append(0)
                continue

            cross = vx1 * vy2 - vy1 * vx2
            dot = vx1 * vx2 + vy1 * vy2

            theta = np.arctan2(cross, dot)
            theta_diffs.append(theta)

        return np.array(theta_diffs)

    def pick_needed_line(self, direct_lines, target_theta):
        """
        pick directline which has target angle
        """
        needed_lines = []
        for direct_line in direct_lines:
            if len(direct_line) < self.cfg.min_length_thresh:
                continue

            ny, nx = np.sin(target_theta), np.cos(target_theta)

            sy, sx = direct_line[0]
            fy, fx = direct_line[-1]

            dot = (fx - sx) * nx + (fy - sy) * ny
            norm = np.hypot(fx - sx, fy - sy)

            if norm == 0:
                continue

            cos = dot / norm

            if abs(cos) >= np.cos(self.cfg.target_theta_tolerance):
                needed_lines.append(direct_line)

        logger.debug(
            f"return {len(needed_lines)} needed linnes and removed {len(direct_lines) - len(needed_lines)}"
        )
        return needed_lines

    # -------------------------
    # Merge
    # -------------------------
    def _split_by_length(
        self, lines: List[List[Tuple[int, int]]]
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """
        セグメントを「他のセグメントとの接続候補として扱うもの
        (connectable)」と「単独でそのまま出力するもの(standalone)」に
        分ける。退化セグメント(点が1つ以下)はここで除外する。

        connect_dist_thresh に対して十分短いセグメントは、端点の方向
        推定が数点のブレに弱く、誤接続の原因になりやすい。そのため
        他のセグメントとマッチングする対象からは外し、standaloneとして
        そのまま最終結果に含める(出力からは除外しない)。
        """
        min_length = self.cfg.connect_dist_thresh * self.cfg.min_segment_length_ratio

        connectable: List[List[Tuple[int, int]]] = []
        standalone: List[List[Tuple[int, int]]] = []

        for line in lines:
            if len(line) < 2:
                continue
            if self._segment_length(line) < min_length:
                standalone.append(line)
            else:
                connectable.append(line)

        logger.debug(
            f"_split_by_length: connectable={len(connectable)}, "
            f"standalone(too short)={len(standalone)}"
        )

        return connectable, standalone

    def _segment_length(self, line: List[Tuple[int, int]]) -> float:
        """セグメントの始点・終点間の直線距離を長さとして使う。"""
        (y0, x0), (y1, x1) = line[0], line[-1]
        return float(np.hypot(y1 - y0, x1 - x0))

    def _collect_endpoints(
        self, segments: List[List[Tuple[int, int]]]
    ) -> List[_Endpoint]:
        """
        各セグメントの両端点を抽出する。
        endpoints[2*k]   = segments[k] の head側端点
        endpoints[2*k+1] = segments[k] の tail側端点
        という固定レイアウトにし、_other_endpoint_idx() で対になる
        端点を定数時間で求められるようにしている。
        """
        endpoints: List[_Endpoint] = []

        for seg_idx, line in enumerate(segments):
            endpoints.append(
                _Endpoint(
                    segment_idx=seg_idx,
                    is_head=True,
                    point=line[0],
                    direction=self._endpoint_direction(line, is_head=True),
                )
            )
            endpoints.append(
                _Endpoint(
                    segment_idx=seg_idx,
                    is_head=False,
                    point=line[-1],
                    direction=self._endpoint_direction(line, is_head=False),
                )
            )

        return endpoints

    def _endpoint_direction(
        self, line: List[Tuple[int, int]], is_head: bool
    ) -> Tuple[float, float]:
        """
        端点における外向きの方向ベクトル(dy, dx、単位ベクトル)を返す。
        端点側からdirection_estimation_window点を見て、その点列を最も
        良く説明する直線の方向(主成分方向)を方向ベクトルとして採用する。
        """
        window = min(self.cfg.direction_estimation_window, len(line))

        if is_head:
            pts = line[:window]
        else:
            pts = list(reversed(line[-window:]))

        return self._fit_outward_direction(pts)

    def _fit_outward_direction(self, pts: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        pts[0] を端点、pts[1:] をセグメント内側へ向かう点列として、
        主成分方向(点列を最小二乗的に最も良く説明する直線の方向)を
        方向ベクトルとして求める。符号は端点側を正方向に揃える。
        """
        if len(pts) < 2:
            return (0.0, 0.0)

        arr = np.asarray(pts, dtype=np.float64)  # (k, 2) = (y, x)
        centroid = arr.mean(axis=0)
        centered = arr - centroid

        if np.allclose(centered, 0.0):
            return (0.0, 0.0)

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        outward = arr[0] - centroid
        if np.dot(outward, direction) < 0.0:
            direction = -direction

        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return (0.0, 0.0)

        direction = direction / norm
        return (float(direction[0]), float(direction[1]))

    def _search_close_pairs(self, endpoints: List[_Endpoint]) -> set:
        if not endpoints:
            return set()

        endpoint_points = [(e.point[1], e.point[0]) for e in endpoints]  # (x, y)
        points_np = np.array(endpoint_points)
        tree = cKDTree(points_np)

        return tree.query_pairs(r=self.cfg.connect_dist_thresh)

    def _filter_pairs(
        self,
        endpoints: List[_Endpoint],
        pairs: set,
    ) -> List[Tuple[int, int]]:
        """
        KDTreeが見つけたペアの中から、実際の接続として採用するものを
        絞り込む。

        1. 同一セグメントの両端点同士を除外する。
        2. 方向の整合性が低い(2本が向き合っていない)ペアを除外する。
        3. 1つの端点が採用できる接続を最大1本に制限する
           (整合度が高いペアを優先する貪欲法)。
        """
        candidates: List[Tuple[int, int]] = []

        for i, j in tqdm(list(pairs), disable=not self.debug):
            if endpoints[i].segment_idx == endpoints[j].segment_idx:
                continue
            if not self._is_direction_consistent(endpoints[i], endpoints[j]):
                continue
            candidates.append((i, j))

        return self._enforce_one_to_one(endpoints, candidates)

    def _is_direction_consistent(self, ep_a: _Endpoint, ep_b: _Endpoint) -> bool:
        """
        2つの端点を接続して良いかを、方向の整合性で判定する。
        端点同士を結ぶベクトルの向きが、両端点それぞれの外向き方向
        とどれだけ揃っているかをcos類似度で同時にチェックする。
        """
        (ya, xa), (yb, xb) = ep_a.point, ep_b.point
        connecting = np.array([yb - ya, xb - xa], dtype=np.float64)
        norm = float(np.linalg.norm(connecting))

        if norm == 0.0:
            return True

        dir_a = np.asarray(ep_a.direction, dtype=np.float64)
        dir_b = np.asarray(ep_b.direction, dtype=np.float64)

        if np.linalg.norm(dir_a) == 0.0 or np.linalg.norm(dir_b) == 0.0:
            return True

        unit = connecting / norm

        align_a = float(np.dot(unit, dir_a))
        align_b = float(np.dot(-unit, dir_b))

        thresh = self.cfg.direction_alignment_cos_thresh
        return align_a >= thresh and align_b >= thresh

    def _enforce_one_to_one(
        self,
        endpoints: List[_Endpoint],
        pairs: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        1つの端点が採用できる接続を最大1本に制限する。
        方向の整合度が高いペアから順に貪欲に採用する。
        """
        scored = [
            (self._pair_alignment_score(endpoints[i], endpoints[j]), i, j)
            for i, j in pairs
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        used: set = set()
        accepted: List[Tuple[int, int]] = []

        for _, i, j in scored:
            if i in used or j in used:
                continue
            accepted.append((i, j))
            used.add(i)
            used.add(j)

        return accepted

    def _pair_alignment_score(self, ep_a: _Endpoint, ep_b: _Endpoint) -> float:
        """ペアの「繋ぎやすさ」のスコア(大きいほど良い接続)。"""
        (ya, xa), (yb, xb) = ep_a.point, ep_b.point
        connecting = np.array([yb - ya, xb - xa], dtype=np.float64)
        norm = float(np.linalg.norm(connecting))

        if norm == 0.0:
            return 2.0

        dir_a = np.asarray(ep_a.direction, dtype=np.float64)
        dir_b = np.asarray(ep_b.direction, dtype=np.float64)
        unit = connecting / norm

        align_a = float(np.dot(unit, dir_a))
        align_b = float(np.dot(-unit, dir_b))

        return align_a + align_b

    def _build_adjacency(
        self,
        n_endpoints: int,
        pairs: List[Tuple[int, int]],
    ) -> dict:
        """採用されたペアから、端点インデックス間の隣接リストを構築する。"""
        adjacency: dict = {i: [] for i in range(n_endpoints)}

        for i, j in pairs:
            adjacency[i].append(j)
            adjacency[j].append(i)

        return adjacency

    def _trace_chain(
        self,
        segments: List[List[Tuple[int, int]]],
        endpoints: List[_Endpoint],
        adjacency: dict,
    ) -> List[List[Tuple[int, int]]]:
        """
        隣接関係(adjacency)に基づき、全セグメントを連結グラフとして辿り、
        連結済みの折れ線群を返す。
        """
        visited = [False] * len(segments)
        connected_lines: List[List[Tuple[int, int]]] = []

        for seg_idx in range(len(segments)):
            if visited[seg_idx]:
                continue
            connected_lines.append(
                self._trace_chain_from(seg_idx, segments, endpoints, adjacency, visited)
            )

        return connected_lines

    def _trace_chain_from(
        self,
        start_seg: int,
        segments: List[List[Tuple[int, int]]],
        endpoints: List[_Endpoint],
        adjacency: dict,
        visited: List[bool],
    ) -> List[Tuple[int, int]]:
        """
        start_seg を起点に、接続グラフを両方向(head側・tail側)に辿り、
        セグメントの点列を順序・向きを揃えて1本の折れ線に結合する。
        """
        visited[start_seg] = True
        points = list(segments[start_seg])  # head -> tail の順

        points = self._extend_chain(
            points,
            2 * start_seg + 1,
            segments,
            endpoints,
            adjacency,
            visited,
            append_to_end=True,
        )
        points = self._extend_chain(
            points,
            2 * start_seg,
            segments,
            endpoints,
            adjacency,
            visited,
            append_to_end=False,
        )

        return points

    def _extend_chain(
        self,
        points: List[Tuple[int, int]],
        from_endpoint_idx: int,
        segments: List[List[Tuple[int, int]]],
        endpoints: List[_Endpoint],
        adjacency: dict,
        visited: List[bool],
        *,
        append_to_end: bool,
    ) -> List[Tuple[int, int]]:
        """
        from_endpoint_idx から接続されている未訪問のセグメントをたどり、
        points の末尾(append_to_end=True)または先頭(False)に
        点列を継ぎ足していく。
        """
        current_idx = from_endpoint_idx

        while True:
            candidates = [
                other
                for other in adjacency.get(current_idx, [])
                if not visited[endpoints[other].segment_idx]
            ]

            if not candidates:
                break

            if len(candidates) > 1:
                logger.debug(
                    "端点 %d に複数の接続候補が残っています。"
                    "最も近いものを採用します。",
                    current_idx,
                )
                candidates.sort(
                    key=lambda other: self._endpoint_distance(
                        endpoints[current_idx], endpoints[other]
                    )
                )

            next_endpoint_idx = candidates[0]
            next_seg = endpoints[next_endpoint_idx].segment_idx
            visited[next_seg] = True

            next_points = list(segments[next_seg])
            if endpoints[next_endpoint_idx].is_head != append_to_end:
                next_points.reverse()

            if append_to_end:
                points = points + next_points
            else:
                points = next_points + points

            current_idx = self._other_endpoint_idx(next_endpoint_idx)

        return points

    def _other_endpoint_idx(self, endpoint_idx: int) -> int:
        """同じセグメントの、もう一方の端点のインデックスを返す。"""
        return endpoint_idx + 1 if endpoint_idx % 2 == 0 else endpoint_idx - 1

    def _endpoint_distance(self, a: _Endpoint, b: _Endpoint) -> float:
        (y1, x1), (y2, x2) = a.point, b.point
        return float(np.hypot(y1 - y2, x1 - x2))

    # -------------------------
    # Erase
    # -------------------------
    def draw_straight_line(
        self, image_shape: Tuple[int, int], lines: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        straight_map = np.zeros(image_shape, dtype=np.uint8)

        for line in lines:
            if len(line) < 2:
                continue
            for i in range(len(line) - 1):
                y1, x1 = line[i]
                y2, x2 = line[i + 1]
                cv2.line(
                    straight_map,
                    (x1, y1),
                    (x2, y2),
                    color=[255],
                    thickness=1,
                )

        kernel = np.ones((3, 3), np.uint8)
        straight_map = cv2.morphologyEx(
            straight_map,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            straight_map,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        ordered_map = np.zeros(image_shape, dtype=np.uint8)

        for contour in contours:
            if len(contour) < 2:
                continue
            for i in range(len(contour) - 1):
                x1, y1 = contour[i][0]
                x2, y2 = contour[i + 1][0]
                cv2.line(
                    ordered_map,
                    (x1, y1),
                    (x2, y2),
                    color=[1],
                    thickness=1,
                )

        logger.debug("completed making map with straight lines")
        return ordered_map

    def remove_noise_line(
        self, binary: np.ndarray, line_image: np.ndarray
    ) -> np.ndarray:
        # remove most part of line noise
        contours, _ = cv2.findContours(
            image=line_image, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE
        )

        half_way = self.remove_inside_contours(binary=binary, contours=contours)

        # remove remaining part of line noise
        contours, _ = cv2.findContours(
            image=half_way, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE
        )

        needless_contours = []
        for contour in contours:
            area = cv2.contourArea(contour=contour)
            if area < self.cfg.min_char_domain:
                needless_contours.append(contour)

        removed = self.remove_inside_contours(
            binary=half_way, contours=needless_contours
        )

        logger.debug(f"remove {len(needless_contours)} needless_contours")

        if self.debug:
            self._visualize_erase(
                binary, line_image, half_way, needless_contours, removed
            )

        return removed

    def remove_inside_contours(self, binary, contours):
        mask = np.zeros_like(binary, dtype=np.uint8)

        for contour in contours:
            cv2.drawContours(
                mask,
                [contour],
                contourIdx=-1,
                color=[255],
                thickness=-1,
            )

        removed = binary.copy()
        removed[mask > 0] = 0

        return removed

    # -------------------------
    # Debug
    # -------------------------

    def _visualize_detection(
        self,
        binary: np.ndarray,
        contours: Sequence[np.ndarray],
        contour_vectors: List[ContourVectors],
        direct_lines: List[List[Tuple[int, int]]],
        needed_lines: List[List[Tuple[int, int]]],
        target_theta: float,
    ) -> None:
        """Detectionフェーズの視覚化"""
        fig, axes = plt.subplots(4, 1, figsize=(8, 8))
        fig.patch.set_facecolor("lightgray")  # ウィンドウ背景をグレーに
        fig.suptitle(f"Detection Phase (Target Theta: {np.rad2deg(target_theta):.0f}°)")

        # 背景用にカラー化
        base_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 1. contours
        img_contours = base_bgr.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
        axes[0].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Contours")

        # 2. contours + contour_vectors
        img_vectors = base_bgr.copy()
        for cv_data in contour_vectors:
            # 視認性のため適度に間引いてベクトル（接線）を描画
            step = max(1, len(cv_data.points) // 10)
            for i in range(0, len(cv_data.points), step):
                y, x = cv_data.points[i]
                vx, vy = cv_data.tangents[i]
                # ベクトルを正規化して描画
                norm = np.hypot(vx, vy)
                if norm > 0:
                    vx, vy = (vx / norm) * 10, (vy / norm) * 10
                    cv2.arrowedLine(
                        img_vectors,
                        (int(x), int(y)),
                        (int(x + vx), int(y + vy)),
                        (255, 0, 0),
                        1,
                    )
        axes[1].imshow(cv2.cvtColor(img_vectors, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Contour Vectors")

        # 3. binary + direct_lines
        img_direct = base_bgr.copy()
        for line in direct_lines:
            for i in range(len(line) - 1):
                y1, x1 = line[i]
                y2, x2 = line[i + 1]
                cv2.line(img_direct, (x1, y1), (x2, y2), (0, 255, 255), 1)
        axes[2].imshow(cv2.cvtColor(img_direct, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. Direct Lines (Before Angle Filter)")

        # 4. binary + needed_lines
        img_needed = base_bgr.copy()
        for line in needed_lines:
            for i in range(len(line) - 1):
                y1, x1 = line[i]
                y2, x2 = line[i + 1]
                cv2.line(img_needed, (x1, y1), (x2, y2), (0, 0, 255), 2)
        axes[3].imshow(cv2.cvtColor(img_needed, cv2.COLOR_BGR2RGB))
        axes[3].set_title("4. Needed Lines (After Angle Filter)")

        for ax in axes.flatten():
            ax.axis("off")
        plt.tight_layout()
        plt.show(block=False)

    def _visualize_merge(self, needed_lines, endpoints, pairs, connected_lines):
        # 縦に4つ並べるレイアウト
        fig, axes = plt.subplots(4, 1, figsize=(8, 16))
        fig.patch.set_facecolor("lightgray")
        fig.suptitle("Merge Phase", fontsize=14)

        # 共通のベース描画（needed_linesをグレーで描画）
        def draw_base(canvas):
            for line in needed_lines:
                for i in range(len(line) - 1):
                    # BGR順に注意 (y, x)
                    cv2.line(
                        canvas,
                        (line[i][1], line[i][0]),
                        (line[i + 1][1], line[i + 1][0]),
                        (100, 100, 100),
                        1,
                    )
            return canvas

        # 1. needed_lines 単体
        canvas1 = draw_base(np.zeros((*self.image_shape, 3), dtype=np.uint8))
        axes[0].imshow(canvas1)
        axes[0].set_title("1. Needed Lines")

        # 2. endpoints + direction + needed_lines
        canvas2 = draw_base(np.zeros((*self.image_shape, 3), dtype=np.uint8))
        for ep in endpoints:
            y, x = ep.point
            # 端点を鮮やかな緑で
            cv2.circle(canvas2, (x, y), 4, (0, 255, 0), -1)
            # 方向ベクトルを鮮やかな赤で
            dy, dx = ep.direction
            cv2.arrowedLine(
                canvas2, (x, y), (int(x + dx * 20), int(y + dy * 20)), (255, 0, 0), 2
            )
        axes[1].imshow(canvas2)
        axes[1].set_title("2. Endpoints & Directions")

        # 3. pairs + needed_lines
        canvas3 = draw_base(np.zeros((*self.image_shape, 3), dtype=np.uint8))
        for i, j in pairs:
            p1, p2 = endpoints[i].point, endpoints[j].point
            # 接続ペアを鮮やかなマゼンタで
            cv2.line(canvas3, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 255), 2)
        axes[2].imshow(canvas3)
        axes[2].set_title("3. Connected Pairs")

        # 4. connected_lines (下地は無しで鮮明に)
        canvas4 = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        cmap = colormaps.get_cmap("hsv")
        for idx, line in enumerate(connected_lines):
            # HSVで色分け
            color = tuple(
                int(c * 255) for c in cmap(idx / max(1, len(connected_lines)))[:3]
            )
            bgr_color = (color[2], color[1], color[0])
            for i in range(len(line) - 1):
                cv2.line(
                    canvas4,
                    (line[i][1], line[i][0]),
                    (line[i + 1][1], line[i + 1][0]),
                    bgr_color,
                    2,
                )
        axes[3].imshow(canvas4)
        axes[3].set_title("4. Final Connected Lines")

        for ax in axes:
            ax.set_facecolor("whitesmoke")
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=False)

    def _visualize_erase(
        self,
        binary: np.ndarray,
        line_image: np.ndarray,
        half_way: np.ndarray,
        needless_contours: list,
        line_removed: np.ndarray,
    ) -> None:
        """Eraseフェーズの視覚化"""
        fig, axes = plt.subplots(4, 1, figsize=(8, 8))
        fig.suptitle("Erase Phase")

        # 1. line_image (入力された消去用マスクの元画像)
        axes[0].imshow(line_image, cmap="gray")
        axes[0].set_title("1. Line Image (Ordered Map)")

        # 2. mask (実際に引き算に使われる塗りつぶし領域)
        # remove_inside_contours と同じ要領で可視化用マスクを再生成
        mask_vis = np.zeros_like(binary, dtype=np.uint8)
        contours, _ = cv2.findContours(
            line_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(mask_vis, contours, -1, [255], -1)
        axes[1].imshow(mask_vis, cmap="gray")
        axes[1].set_title("2. Filled Mask")

        # 3. half_way + needless_contours
        img_halfway = cv2.cvtColor(half_way, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(
            img_halfway, needless_contours, -1, (255, 0, 0), -1
        )  # 赤で塗りつぶし
        axes[2].imshow(cv2.cvtColor(img_halfway, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"3. Halfway + Needless (Area < {self.cfg.min_char_domain})")

        # 4. line_removed
        axes[3].imshow(line_removed, cmap="gray")
        axes[3].set_title("4. Final Line Removed")

        for ax in axes.flatten():
            ax.axis("off")
        plt.tight_layout()

        # 最後のプロットなのでブロッキングしてユーザーに見せる
        plt.show(block=True)
