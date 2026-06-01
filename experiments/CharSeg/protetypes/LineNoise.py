from typing import List, Tuple
from dataclasses import dataclass

from tqdm import tqdm
from scipy.spatial import cKDTree
import cv2
import numpy as np

from experiments.CharSeg.protetypes.context import Context


@dataclass
class CNRConfig:
    binary_threshhold: int = 128
    open_kernel_size: int = 3
    erosion_kernel_size: int = 1
    erosion_iterations: int = 0
    # line remove
    line_theta_deg: float = 0.0
    line_theta_tolerance_deg: float = 5.0
    line_min_length: int = 20
    # contour vector
    cont_nighr_len: int = 10
    max_theta_thresh: np.float16 = np.deg2rad(3, dtype=np.float16)
    # picking vectors
    min_length_thresh: np.float16 = np.float16(20)
    target_theta: np.float16 = np.deg2rad(0, dtype=np.float16)
    target_theta_tolerance: np.float16 = np.deg2rad(3, dtype=np.float16)
    # line connection
    connect_dist_thresh: np.uint8 = np.uint8(100)


class ContourNoiseRemover:
    cfg = CNRConfig()

    def process(self, context: Context):
        img = self.load_image(context.image_path)

        binary = self.preprocess(img)

        contours = self.detect_contours(binary)

        cont_vecs, contours = self.arrange_contour_vectors2(contours=contours)

        direct_lines = self.detect_direct_line(cont_vecs=cont_vecs, contours=contours)

        direct_lines = self.pick_needed_line(
            direct_lines=direct_lines, target_theta=self.cfg.target_theta
        )

        connected_lines = self.connect_splitted_line(
            straight_lines=direct_lines, binary_shape=binary.shape
        )

        line_map = self.draw_staraight_line(
            binary=binary, straight_lines=connected_lines
        )

        self.visualize_result(
            img=img,
            binary=binary,
            contours=contours,
            cont_vecs=cont_vecs,
            horizontal_map=line_map,
        )

        context.line_removed = self.remove_noise_line(binary=binary, line_map=line_map)

        # self.visualize_after_removed(
        #     binary=binary,
        #     line_map=line_map,
        #     removed=,
        # )

    def load_image(self, image_path: str):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"failed to load image: {image_path}")

        return img

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        open_kernel = np.ones(
            (self.cfg.open_kernel_size, self.cfg.open_kernel_size), np.uint8
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        if self.cfg.erosion_iterations > 0:
            erosion_kernel = np.ones(
                (self.cfg.erosion_kernel_size, self.cfg.erosion_kernel_size)
            )

            binary = cv2.erode(
                binary,
                erosion_kernel,
                iterations=self.cfg.erosion_iterations,  # for some times
            )

        return binary

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

        print(f"arranged tangent vectors about {len(valid_contours)} valid_contours")
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

        print(f"detcted {len(direct_lines)} direct_lines")
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

        print(
            f"return {len(needed_lines)} needed linnes and removed {len(direct_lines) - len(needed_lines)}"
        )
        return needed_lines

    def connect_splitted_line(self, straight_lines, binary_shape):

        line_map = np.zeros(binary_shape, dtype=np.uint8)

        # =========================
        # draw original line segments
        # =========================
        for line in straight_lines:

            if len(line) < 2:
                continue

            for i in range(len(line) - 1):

                y1, x1 = line[i]
                y2, x2 = line[i + 1]

                cv2.line(
                    line_map,
                    (x1, y1),
                    (x2, y2),
                    color=255,
                    thickness=1,
                )

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

                idxA, pA = endpoints[i]
                idxB, pB = endpoints[j]

                if idxA == idxB:
                    continue

                y1, x1 = pA
                y2, x2 = pB

                cv2.line(
                    line_map,
                    (x1, y1),
                    (x2, y2),
                    color=255,
                    thickness=1,
                )

        # =========================
        # morphology close
        # =========================
        kernel = np.ones((3, 3), np.uint8)

        line_map = cv2.morphologyEx(
            line_map,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # =========================
        # contourize connected map
        # =========================
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

        print(f"connected lines : {len(straight_lines)} -> {len(connected_lines)}")

        return connected_lines

    def draw_staraight_line(self, binary: np.ndarray, straight_lines):

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
                    color=255,
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
                    color=1,
                    thickness=1,
                )

        print("completed making map with straight lines")

        return ordered_map

    def remove_noise_line(self, binary, line_map):

        contours, hierarchy = cv2.findContours(
            line_map,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE,
        )

        mask = np.zeros_like(binary, dtype=np.uint8)

        for contour in contours:

            cv2.drawContours(
                mask,
                [contour],
                contourIdx=-1,
                color=255,
                thickness=-1,
            )

        removed = binary.copy()
        removed[mask > 0] = 0

        return removed

    def visualize_result(
        self,
        img: np.ndarray,
        binary: np.ndarray,
        contours: List[np.ndarray],
        cont_vecs: List,
        horizontal_map: np.ndarray,
        scale: int = 8,
    ):
        """
        visualize:
        1. binary
        2. contour only
        3. contour + tangent vector + detected horizontal pixels
        """

        import matplotlib.pyplot as plt

        # =========================
        # image1: binary
        # =========================
        binary_vis = binary.copy()

        # =========================
        # image2: contour only
        # =========================
        contour_vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(
            contour_vis,
            contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=1,
        )

        # =========================
        # image3: full visualization
        # =========================
        vis = contour_vis.copy()

        # tangent vector描画
        for i, tangent_vecs in enumerate(cont_vecs):

            contour = contours[i]

            for j, (vx, vy) in enumerate(tangent_vecs):

                x, y = contour[j % len(contour)][0]

                norm = np.hypot(vx, vy)

                if norm == 0:
                    continue

                # normalize
                vx = vx / norm
                vy = vy / norm

                ex = int(x + vx * scale)
                ey = int(y + vy * scale)

                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (ex, ey),
                    (255, 0, 0),
                    1,
                    tipLength=0.2,
                )

        # horizontal pixel を赤で重ねる
        vis[horizontal_map > 0] = (0, 0, 255)

        # =========================
        # matplotlib表示
        # =========================

        dpi = 100
        h, w = binary.shape

        fig_w = w / dpi
        fig_h = h / dpi

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(fig_w, fig_h * 3),
            dpi=dpi,
        )

        # binary
        axes[0].imshow(binary_vis, cmap="gray")
        axes[0].set_title("Binary")
        axes[0].axis("off")

        # contour only
        axes[1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Contour")
        axes[1].axis("off")

        # full visualization
        axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Contour + Tangent + Detection")
        axes[2].axis("off")

        plt.subplots_adjust(
            left=0,
            right=1,
            top=0.98,
            bottom=0.02,
            hspace=0.1,
        )

        plt.show()

    def visualize_after_removed(
        self,
        binary: np.ndarray,
        line_map: np.ndarray,
        removed: np.ndarray,
    ):

        import matplotlib.pyplot as plt

        dpi = 100
        h, w = binary.shape

        fig_w = w / dpi
        fig_h = h / dpi

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(fig_w, fig_h * 3),
            dpi=dpi,
        )

        # =========================
        # image1: original binary
        # =========================
        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Binary")
        axes[0].axis("off")

        # =========================
        # image2: detected contours with different colors
        # =========================
        contours, _ = cv2.findContours(
            line_map.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        color_vis = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(5)

        for contour in contours:

            color = rng.integers(0, 255, size=3).tolist()

            cv2.drawContours(
                color_vis,
                [contour],
                contourIdx=-1,
                color=color,
                thickness=1,
            )

        axes[1].imshow(cv2.cvtColor(color_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Detected Lines")
        axes[1].axis("off")

        # =========================
        # image3: removed result
        # =========================
        axes[2].imshow(removed, cmap="gray")
        axes[2].set_title("After Line Removal")
        axes[2].axis("off")

        plt.subplots_adjust(
            left=0,
            right=1,
            top=0.98,
            bottom=0.02,
            hspace=0.1,
        )

        plt.show()
