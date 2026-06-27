import logging
from typing import List, Tuple, cast
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import cv2
import numpy as np

from CharSeg.context import Context

logger = logging.getLogger(__name__)


@dataclass
class LNRConfig:

    #
    binary_threshhold: int = 128
    open_kernel_size: int = 3
    erosion_kernel_size: int = 1
    erosion_iterations: int = 0

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
    # 改善5: connect_dist_threshに対する比率。これより短いセグメントは
    # 他のセグメントとの接続候補(マッチング対象)から外し、単独の線として扱う。
    min_segment_length_ratio: float = 0.3
    # 改善2: 端点の方向推定に使う点数(端点から何点を見て方向を推定するか)。
    direction_estimation_window: int = 5
    # 改善1: 接続を許可する最低のcos類似度(0~1)。1に近いほど厳密に
    # 「正面を向いて並んでいる」場合のみ接続を許可する。
    direction_alignment_cos_thresh: float = 0.7
    #
    min_char_domain: int = 100


@dataclass
class _Endpoint:
    """直線セグメントの端点情報。

    direction は端点における外向きの方向ベクトル(dy, dx、単位ベクトル)。
    端点付近の複数点から主成分方向として推定する(改善2)。
    _filter_pairsでの方向整合性チェック(改善1)・一対一制約(改善4)で使う。
    """

    segment_idx: int
    is_head: bool
    point: Tuple[int, int]
    direction: Tuple[float, float]


class LineNoiseRemover:
    cfg = LNRConfig()

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, context: Context):
        logger.debug("ホワイトボードのマス目の線を取り除きます。")
        binary = context.preprocessed

        if binary is None:
            raise ValueError("contextのpreorocessedがNoneです。")

        context.line_removed = self.remove_line_angle(
            binary=binary, target_theta=self.cfg.target_theta_horizontal
        )
        context.line_removed = self.remove_line_angle(
            binary=context.line_removed, target_theta=self.cfg.target_theta_vertical
        )

    def remove_line_angle(self, binary: np.ndarray, target_theta):
        contours = self.detect_contours(binary=binary)

        cont_vecs, contours = self.arrange_contour_vectors2(contours=contours)

        direct_lines = self.detect_direct_line(cont_vecs=cont_vecs, contours=contours)

        needed_lines = self.pick_needed_line(
            direct_lines=direct_lines, target_theta=target_theta
        )
        connected_lines = self.connect_splitted_line(
            straight_lines=needed_lines,
        )
        line_map = self.draw_staraight_line(
            binary=binary,
            straight_lines=connected_lines,
        )

        line_removed = self.remove_noise_line(binary=binary, line_map=line_map)

        if self.debug:
            self.visualize(
                binary=binary,
                contours=contours,
                cont_vecs=cont_vecs,
                direct_lines=direct_lines,
                line_map=line_map,
                line_removed=line_removed,
            )

        return line_removed

    def detect_contours(self, binary):

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,  # only outside edges
            cv2.CHAIN_APPROX_NONE,
        )

        return contours

    def arrange_contour_vectors2(self, contours: np.ndarray):
        """
        calculate tangent vector on contour more simply
        """
        contour_vectors = []
        valid_contours = []
        cont_neighr_len = self.cfg.cont_nighr_len

        for contour in contours:
            length = len(contour)
            if length < 2 * cont_neighr_len + 1:
                continue

            tangent_vectors = []

            start = 0
            fin = length

            for i in range(start, fin):
                idx1 = (i - cont_neighr_len) % length
                idx2 = (i + cont_neighr_len) % length
                x1, y1 = contour[idx1][0]
                x2, y2 = contour[idx2][0]
                tangent_vectors.append((x2 - x1, y2 - y1))

            contour_vectors.append(tangent_vectors)
            valid_contours.append(contour)

        logger.debug(
            f"arranged tangent vectors about {len(valid_contours)} valid_contours"
        )
        # the idx of contour corresponed to tangent_vector's
        return contour_vectors, valid_contours

    def detect_direct_line(
        self, cont_vecs: List[np.ndarray], contours: List[np.ndarray]
    ):
        """
        return direct line as like a cv2.findContour()
        """
        direct_lines = []

        for i, (tan_vecs, contour) in enumerate(zip(cont_vecs, contours)):
            direct_line = []

            theta_diffs = self.vector_difference_theta(tan_vecs=tan_vecs)
            length = len(theta_diffs)

            for j in range(length):
                idx = j % length

                if abs(theta_diffs[idx]) <= self.cfg.max_theta_thresh:
                    x, y = contour[idx][0]
                    direct_line.append((y, x))
                elif len(direct_line) > 2:
                    direct_lines.append(direct_line)
                    direct_line = []

            if len(direct_line) > 2:
                direct_lines.append(direct_line)

        logger.debug(f"detcted {len(direct_lines)} direct_lines")
        return direct_lines

    def vector_difference_theta(self, tan_vecs: np.ndarray):
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

            cross = vx1 * vy2 - vx2 * vy1
            cross = vx1 * vy2 - vy1 * vx2
            dot = vx1 * vx2 + vy1 * vy2

            theta = np.arctan2(cross, dot)
            theta_diffs.append(theta)

        return np.array(theta_diffs)

    def pick_needed_line(self, direct_lines, target_theta: np.float16):
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

    def connect_splitted_line(
        self,
        straight_lines: List[List[Tuple[int, int]]],
    ) -> List[List[Tuple[int, int]]]:
        """
        分断された直線セグメント群を、端点同士の近接関係に基づいて
        実際の点列レベルで連結する。

        パイプライン:
            1. _split_by_length    : 接続候補(connectable)と単独出力
                                      (standalone)にセグメントを分ける(改善5)
            2. _collect_endpoints  : 各セグメントの両端点の方向ベクトルを
                                      複数点から推定する(改善2)
            3. _search_close_pairs : KDTreeで近接ペアを探索(既存ロジックのまま)
            4. _filter_pairs       : 方向の整合性が低いペアを除外し(改善1)、
                                      端点ごとに採用1本までに制限する(改善4)
            5. _build_adjacency    : 採用ペアから端点間の隣接関係を構築
            6. _trace_chain        : 元のセグメントの点列を、順序・向きを
                                      揃えて直接連結する(改善3)

        旧実装は cv2.line による描画 → morphologyEx → findContours という
        手順で線を再構成していたが、findContoursは線の輪郭(縁)を辿るため
        1本の線の片側を辿って反対側を辿って戻ってくる往復点列になり、
        座標が破壊されていた。本実装はラスタライズを経由せず、元のセグ
        メントの点列を直接つなぎ合わせることでこれを避ける。(改善3)
        """
        valid_lines = [line for line in straight_lines if len(line) >= 2]

        if not valid_lines:
            return []

        connectable, standalone = self._split_by_length(valid_lines)

        if not connectable:
            return standalone

        endpoints = self._collect_endpoints(connectable)

        raw_pairs = self._search_close_pairs(endpoints)
        pairs = self._filter_pairs(endpoints, raw_pairs)

        adjacency = self._build_adjacency(len(endpoints), pairs)

        visited = [False] * len(connectable)
        connected_lines: List[List[Tuple[int, int]]] = list(standalone)

        for seg_idx in range(len(connectable)):
            if visited[seg_idx]:
                continue
            connected_lines.append(
                self._trace_chain(seg_idx, connectable, endpoints, adjacency, visited)
            )

        logger.debug(
            f"connected lines : {len(straight_lines)} -> {len(connected_lines)}"
            f" (standalone: {len(standalone)})"
        )

        return connected_lines

    # ------------------------------------------------------------------
    # セグメント選別 (改善5)
    # ------------------------------------------------------------------
    def _split_by_length(
        self, straight_lines: List[List[Tuple[int, int]]]
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """
        セグメントを「他のセグメントとの接続候補として扱うもの
        (connectable)」と「単独でそのまま出力するもの(standalone)」に
        分ける。

        connect_dist_thresh に対して十分短いセグメントは、端点の方向
        推定(改善2)が数点のブレに弱く、誤接続(改善1のフィルタを偶然
        通過してしまう)の原因になりやすい。そのため他のセグメントと
        マッチングする対象からは外し、standaloneとしてそのまま
        最終結果に含める(出力からは除外しない)。
        """
        min_length = self.cfg.connect_dist_thresh * self.cfg.min_segment_length_ratio

        connectable: List[List[Tuple[int, int]]] = []
        standalone: List[List[Tuple[int, int]]] = []

        for line in straight_lines:
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

    # ------------------------------------------------------------------
    # 端点・方向ベクトルの抽出
    # ------------------------------------------------------------------
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

        改善2: 端の1点だけでなく、端点側から
        direction_estimation_window 点を見て、その点列を最も良く説明する
        直線の方向(主成分方向)を方向ベクトルとして採用する。
        端の1点だけに比べ、ノイズや量子化誤差に強い推定になる。
        """
        window = min(self.cfg.direction_estimation_window, len(line))

        if is_head:
            # pts[0]が端点、以降は内側へ向かう点列
            pts = line[:window]
        else:
            pts = list(reversed(line[-window:]))

        return self._fit_outward_direction(pts)

    def _fit_outward_direction(self, pts: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        pts[0] を端点、pts[1:] をセグメント内側へ向かう点列として、
        主成分方向(点列を最小二乗的に最も良く説明する直線の方向)を
        方向ベクトルとして求める。

        主成分方向は符号が不定(±どちらも同じ直線を表す)ため、
        端点(pts[0])が他の点の重心から見て正方向にあるように
        符号を揃え、「外向き」のベクトルにする。
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

    # ------------------------------------------------------------------
    # KDTreeによる近接探索(既存ロジックのまま)
    # ------------------------------------------------------------------
    def _search_close_pairs(self, endpoints: List[_Endpoint]) -> set[Tuple[int, int]]:
        if not endpoints:
            return set()

        endpoint_points = [(e.point[1], e.point[0]) for e in endpoints]  # (x, y)
        points_np = np.array(endpoint_points)
        tree = cKDTree(points_np)

        return tree.query_pairs(r=self.cfg.connect_dist_thresh)

    # ------------------------------------------------------------------
    # 採用ペアの絞り込み (改善1, 改善4)
    # ------------------------------------------------------------------
    def _filter_pairs(
        self,
        endpoints: List[_Endpoint],
        pairs: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        KDTreeが見つけたペアの中から、実際の接続として採用するものを
        絞り込む。

        1. 同一セグメントの両端点同士を除外する。
        2. 改善1: 方向の整合性が低い(2本が向き合っていない)ペアを除外する。
        3. 改善4: 1つの端点が採用できる接続を最大1本に制限する
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
        改善1: 2つの端点を接続して良いかを、方向の整合性で判定する。

        端点同士を結ぶベクトルの向きが、両端点それぞれの外向き方向
        (_endpoint_direction)とどれだけ揃っているかを見る。
        本来1本だった線が分断された場合、両端点は「お互いに向き合う形」
        で、かつ「結ぶ方向が元の線の延長線上」になるはずなので、
        この2条件をcos類似度で同時にチェックする。
        """
        (ya, xa), (yb, xb) = ep_a.point, ep_b.point
        connecting = np.array([yb - ya, xb - xa], dtype=np.float64)
        norm = float(np.linalg.norm(connecting))

        if norm == 0.0:
            # 端点が完全に重なっている場合は許容する
            return True

        dir_a = np.asarray(ep_a.direction, dtype=np.float64)
        dir_b = np.asarray(ep_b.direction, dtype=np.float64)

        if np.linalg.norm(dir_a) == 0.0 or np.linalg.norm(dir_b) == 0.0:
            # 方向が推定できない(極端に短い等)場合は他の条件に委ねる
            return True

        unit = connecting / norm

        # ep_aの外向き方向が、ep_bへ向かう方向とどれだけ揃っているか
        align_a = float(np.dot(unit, dir_a))
        # ep_bの外向き方向が、ep_aへ向かう方向とどれだけ揃っているか
        align_b = float(np.dot(-unit, dir_b))

        thresh = self.cfg.direction_alignment_cos_thresh
        return align_a >= thresh and align_b >= thresh

    def _enforce_one_to_one(
        self,
        endpoints: List[_Endpoint],
        pairs: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        改善4: 1つの端点が採用できる接続を最大1本に制限する。

        方向の整合度が高い(=より「繋ぐべき」と確信できる)ペアから
        順に貪欲に採用し、すでに使われた端点を含むペアはスキップする。
        これにより、1つの端点が複数の端点へ同時に接続される
        分岐(Y字・X字状の誤接続)を防ぐ。
        """
        scored = [
            (self._pair_alignment_score(endpoints[i], endpoints[j]), i, j)
            for i, j in pairs
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        used: set[int] = set()
        accepted: List[Tuple[int, int]] = []

        for _, i, j in scored:
            if i in used or j in used:
                continue
            accepted.append((i, j))
            used.add(i)
            used.add(j)

        return accepted

    def _pair_alignment_score(self, ep_a: _Endpoint, ep_b: _Endpoint) -> float:
        """
        ペアの「繋ぎやすさ」のスコア(大きいほど良い接続)。
        _is_direction_consistentと同じ2つのcos類似度の合計を使う。
        """
        (ya, xa), (yb, xb) = ep_a.point, ep_b.point
        connecting = np.array([yb - ya, xb - xa], dtype=np.float64)
        norm = float(np.linalg.norm(connecting))

        if norm == 0.0:
            return 2.0  # 完全に重なっている場合は最優先で採用する

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
    ) -> dict[int, list[int]]:
        """採用されたペアから、端点インデックス間の隣接リストを構築する。"""
        adjacency: dict[int, list[int]] = {i: [] for i in range(n_endpoints)}

        for i, j in pairs:
            adjacency[i].append(j)
            adjacency[j].append(i)

        return adjacency

    def _other_endpoint_idx(self, endpoint_idx: int) -> int:
        """同じセグメントの、もう一方の端点のインデックスを返す。"""
        return endpoint_idx + 1 if endpoint_idx % 2 == 0 else endpoint_idx - 1

    def _endpoint_distance(self, a: _Endpoint, b: _Endpoint) -> float:
        (y1, x1), (y2, x2) = a.point, b.point
        return float(np.hypot(y1 - y2, x1 - x2))

    # ------------------------------------------------------------------
    # 点列の直接連結 (改善3の本体)
    # ------------------------------------------------------------------
    def _trace_chain(
        self,
        start_seg: int,
        segments: List[List[Tuple[int, int]]],
        endpoints: List[_Endpoint],
        adjacency: dict[int, list[int]],
        visited: List[bool],
    ) -> List[Tuple[int, int]]:
        """
        start_seg を起点に、接続グラフを両方向(head側・tail側)に辿り、
        セグメントの点列を順序・向きを揃えて1本の折れ線に結合する。
        """
        visited[start_seg] = True
        points = list(segments[start_seg])  # head -> tail の順

        # tail側から後ろへ伸ばす
        points = self._extend_chain(
            points,
            2 * start_seg + 1,
            segments,
            endpoints,
            adjacency,
            visited,
            append_to_end=True,
        )
        # head側から前へ伸ばす
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
        adjacency: dict[int, list[int]],
        visited: List[bool],
        *,
        append_to_end: bool,
    ) -> List[Tuple[int, int]]:
        """
        from_endpoint_idx から接続されている未訪問のセグメントをたどり、
        points の末尾(append_to_end=True)または先頭(False)に
        点列を継ぎ足していく。

        ループに戻ってきた場合は、戻り先のセグメントが既にvisited済みの
        ため候補から除外され、無限ループにはならない(輪が完全に閉じず、
        わずかな隙間が残る形で停止する)。

        _filter_pairsの一対一制約(改善4)により、通常は各端点の接続候補は
        最大1本になっている。それでも複数残ってしまった場合に備え、
        距離が最も近いものを採用するフォールバックを残している。
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
                    "端点 %d に複数の接続候補が残っています"
                    "(一対一制約後は通常発生しません)。最も近いものを採用します。",
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

    def draw_staraight_line(self, binary: np.ndarray, straight_lines) -> np.ndarray:

        straight_map = np.zeros_like(binary, dtype=np.uint8)

        for straight_line in straight_lines:

            if len(straight_line) < 2:
                continue

            for i in range(len(straight_line) - 1):

                y1, x1 = straight_line[i]
                y2, x2 = straight_line[i + 1]

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

        ordered_map = np.zeros_like(binary, dtype=np.uint8)

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

    def remove_noise_line(self, binary: np.ndarray, line_map: np.ndarray):

        # remove most part of line noise
        contours, _ = cv2.findContours(
            image=line_map, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE
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

    def visualize(
        self,
        binary: np.ndarray,
        contours: List[np.ndarray],
        cont_vecs: List,
        direct_lines: List,
        line_map: np.ndarray,
        line_removed: np.ndarray,
        scale: int = 8,
    ) -> None:
        if not self.debug:
            return

        import matplotlib.pyplot as plt

        contour_vis = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

        cv2.drawContours(
            contour_vis,
            contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
        )

        # ① Contours + Tangent Vectors
        tangent_vis = contour_vis.copy()

        for contour, tangent_vecs in zip(contours, cont_vecs):
            for j, (vx, vy) in enumerate(tangent_vecs):
                x, y = contour[j % len(contour)][0]

                norm = np.hypot(vx, vy)
                if norm == 0:
                    continue

                vx = vx / norm
                vy = vy / norm

                ex = int(x + vx * scale)
                ey = int(y + vy * scale)

                cv2.arrowedLine(
                    tangent_vis,
                    (x, y),
                    (ex, ey),
                    (255, 0, 255),  # マゼンタ・太め
                    1,
                    tipLength=0.3,
                )

        # ② Binary + Direct Lines（選別前の生データ）
        needed_vis = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

        for line in direct_lines:
            if len(line) < 2:
                continue
            for i in range(len(line) - 1):
                y1, x1 = line[i]
                y2, x2 = line[i + 1]
                cv2.line(
                    needed_vis,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),  # 赤・太め
                    2,
                )

        # ③ Detected Line Map を Binary に重ねる
        line_map_arr = cast(np.ndarray, line_map)

        line_map_vis = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
        line_map_mask = (line_map > 0).astype(np.uint8)

        # 線を膨張させて視認性を上げる（元のpixel幅が1のため）
        thick_kernel = np.ones((3, 3), np.uint8)
        line_map_mask = cv2.dilate(line_map_mask, thick_kernel, iterations=1)
        line_map_mask = (line_map_arr > 0).astype(np.uint8)

        line_map_vis[line_map_mask > 0] = (0, 255, 0)  # 緑・太め

        dpi = 100
        h, w = binary.shape

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(w / dpi, h / dpi * 4),
            dpi=dpi,
        )

        # ① 接ベクトル計算
        axes[0].imshow(cv2.cvtColor(tangent_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Contours + Tangent Vectors")
        axes[0].axis("off")

        # ② 直線検出（選別前・Binaryに重ねて表示）
        axes[1].imshow(cv2.cvtColor(needed_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Binary + Direct Lines (before selection)")
        axes[1].axis("off")

        # ③ 直線連結（Binary に重ねて表示）
        axes[2].imshow(cv2.cvtColor(line_map_vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Detected Line Map (overlay on Binary)")
        axes[2].axis("off")

        # ④ 線除去
        axes[3].imshow(line_removed, cmap="gray")
        axes[3].set_title("After Line Removal")
        axes[3].axis("off")

        plt.subplots_adjust(
            left=0,
            right=1,
            top=0.98,
            bottom=0.02,
            hspace=0.10,
        )

        plt.show()
