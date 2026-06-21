import logging
from typing import List, cast
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import cv2
import numpy as np

from experiments.CharSeg.protetypes.context import Context

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
    #
    min_char_domain: int = 100


class LineNoiseRemover:
    cfg = LNRConfig()

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, context: Context):
        logging.info("ホワイトボードのマス目の線を取り除きます。")
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
            binary_shape=binary.shape,
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

    def is_closed_contour(self, contour: np.ndarray, dist_thresh: float = 1.5) -> bool:
        """
        I didn't notice cv2.findContour returns only closed contour
        """
        if len(contour) < 2:
            return False

        sx, sy = contour[0][0]
        fx, fy = contour[-1][0]

        dist = np.hypot(fx - sx, fy - sy)

        return dist <= dist_thresh

    def arrange_contour_vectors(self, contours: np.ndarray):
        """
        calculate tangent vector on contour
        """
        contour_vectors = []
        for contour in contours:

            small_vectors = []
            px, py = contour[0][0]

            for i in range(1, len(contour), 1):
                cx, cy = contour[i][0]
                small_vectors.append((cx - px, cy - py))
                px, py = cx, cy
            fx, fy = contour[0][0]
            small_vectors.append((fx - px, fy - py))

            length = len(small_vectors)
            tangent_vectors = []

            for i in range(length):
                sum_x, sum_y = 0, 0

                for j in range(self.cfg.cont_nighr_len * 2 + 1):
                    half = self.cfg.cont_nighr_len
                    idx = (i + j - half) % length
                    sum_x += small_vectors[idx][0]
                    sum_y += small_vectors[idx][1]
                tangent_vectors.append((sum_x, sum_y))

            contour_vectors.append(tangent_vectors)

        return contour_vectors

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
            is_closed = self.is_closed_contour(contour=contour)

            if is_closed:
                start = 0
                fin = length
            else:
                start = cont_neighr_len
                fin = length - cont_neighr_len

            for i in range(start, fin):
                idx1 = (i - cont_neighr_len) % length
                idx2 = (i + cont_neighr_len) % length
                x1, y1 = contour[idx1][0]
                x2, y2 = contour[idx2][0]
                tangent_vectors.append((x2 - x1, y2 - y1))

            if not is_closed:
                start_vecs = []
                for i in range(cont_neighr_len):
                    x1, y1 = contour[i][0]
                    x2, y2 = contour[i + cont_neighr_len][0]
                    start_vecs.append((x2 - x1, y2 - y1))

                tangent_vectors = start_vecs + tangent_vectors

                fin_vecs = []
                for i in range(1, cont_neighr_len + 1):
                    x1, y1 = contour[-i][0]
                    x2, y2 = contour[-i - cont_neighr_len][0]
                    fin_vecs.append((x2 - x1, y2 - y1))

                tangent_vectors = tangent_vectors + fin_vecs

            contour_vectors.append(tangent_vectors)
            valid_contours.append(contour)

        logging.info(
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

            is_closed_contour = self.is_closed_contour(contour=contour)

            theta_diffs = self.vector_difference_theta(
                tan_vecs=tan_vecs, is_closed=is_closed_contour
            )
            length = len(theta_diffs)

            for i in range(length):
                idx = i % length

                if abs(theta_diffs[idx]) <= self.cfg.max_theta_thresh:
                    x, y = contour[idx][0]
                    direct_line.append((y, x))
                elif len(direct_line) > 2:
                    direct_lines.append(direct_line)
                    direct_line = []

                if len(direct_line) > 2:
                    direct_lines.append(direct_line)

        logging.info(f"detcted {len(direct_lines)} direct_lines")
        return direct_lines

    def vector_difference_theta(self, tan_vecs: np.ndarray, is_closed: bool = False):
        """
        the difference theta1 and theta2 as a vector difference
        """
        theta_diffs = []
        length = len(tan_vecs)
        for i in range(length):

            if not is_closed and i == length - 1:
                continue

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

        logging.info(
            f"return {len(needed_lines)} needed linnes and removed {len(direct_lines) - len(needed_lines)}"
        )
        return needed_lines

    def connect_splitted_line(self, straight_lines, binary_shape):

        line_map = np.zeros(binary_shape, dtype=np.uint8)

        # # =========================
        # # draw original line segments
        # # =========================
        # for line in straight_lines:

        #     if len(line) < 2:
        #         continue

        #     for i in range(len(line) - 1):

        #         y1, x1 = line[i]
        #         y2, x2 = line[i + 1]

        #         cv2.line(
        #             img=line_map,
        #             pt1=(x1, y1),
        #             pt2=(x2, y2),
        #             color=[255],
        #             thickness=1,
        #         )

        # =========================
        # collect endpoints
        # =========================
        endpoints = []
        endpoint_points = []

        for idx, line in enumerate(straight_lines):

            if len(line) < 2:
                continue

            p1 = line[0]
            p2 = line[-1]

            endpoints.append((idx, p1))
            endpoints.append((idx, p2))

            endpoint_points.append((p1[1], p1[0]))
            endpoint_points.append((p2[1], p2[0]))

        # =========================
        # KDTree nearest search
        # =========================
        if len(endpoint_points) > 0:

            points_np = np.array(endpoint_points)

            tree = cKDTree(points_np)

            pairs = tree.query_pairs(r=self.cfg.connect_dist_thresh)

            for i, j in tqdm(list(pairs)):

                idxA, (y1, x1) = endpoints[i]
                idxB, (y2, x2) = endpoints[j]

                if idxA == idxB:
                    continue

                cv2.line(
                    line_map,
                    (x1, y1),
                    (x2, y2),
                    color=[255],
                    thickness=1,
                )

        kernel = np.ones((3, 3), np.uint8)

        line_map = cv2.morphologyEx(
            line_map,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            line_map,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        connected_lines = []

        for contour in contours:

            if len(contour) < 2:
                continue

            line = []

            for pt in contour:
                x, y = pt[0]
                line.append((y, x))

            connected_lines.append(line)

        logging.info(
            f"connected lines : {len(straight_lines)} -> {len(connected_lines)}"
        )

        return connected_lines

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

        logging.info("completed making map with straight lines")

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

        logging.info(f"remove {len(needless_contours)} needless_contours")

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
