from typing import List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CNRConfig:
    binary_threshhold: int = 128
    open_kernel_size: int = 3
    erosion_kernel_size: int = 1
    erosion_iterations: int = 0
    # input/output
    input_path: str = "photos/sample/cells/toriten.jpeg"
    # contour vector
    cont_nighr_len: int = 2
    max_theta_thresh: np.float16 = np.pi / np.float16(180)
    min_length_thresh: np.float16 = np.float16(100)


class ContourNoiseRemover:
    def __init__(self, config: CNRConfig):
        self.cfg = config

    def run(self, image_path: str):
        img = self.load_image(image_path)

        binary = self.preprocess(img)

        contours = self.detect_contours(binary)

        cont_vecs = self.arrange_contour_vectors2(contours=contours)

        direct_lines = self.detect_direct_line(
            binary=binary, cont_vecs=cont_vecs, contours=contours
        )

        direct_lines = self.rm_needless_line(direct_lines=direct_lines, horizontal=True)

        line_map = self.draw_staraight_line(binary=binary, straight_lines=direct_lines)

        self.visualize_result(
            img=img,
            binary=binary,
            contours=contours,
            cont_vecs=cont_vecs,
            horizontal_map=line_map,
        )

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
            cv2.RETR_EXTERNAL,  # only outside edges
            cv2.CHAIN_APPROX_NONE,
        )

        return contours

    def is_closed_contour(self, contour: np.ndarray, dist_thresh: float = 1.5) -> bool:
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

            contour_vectors.append(tangent_vectors)

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

        return contour_vectors

    def detect_direct_line(
        self, binary: np.ndarray, cont_vecs: np.ndarray, contours: np.ndarray
    ):
        direct_lines = []

        for i, (tan_vecs, contour) in enumerate(zip(cont_vecs, contours)):
            direct_line = []

            theta_diffs = self.vector_difference_theta(tan_vecs=tan_vecs)
            length = len(theta_diffs)
            for i in range(length):
                idx = i % length

                if abs(theta_diffs[idx]) <= self.cfg.max_theta_thresh:
                    x, y = contour[idx][0]
                    direct_line.append((y, x))
            direct_lines.append(direct_line)

        return direct_lines

    def rm_needless_line(self, direct_lines, horizontal: bool = True):
        needed_lines = []
        for direct_line in direct_lines:
            sx, sy = direct_line[0]
            fx, fy = direct_line[-1]

            if horizontal:
                dot = fx - sx
            else:
                dot = sy - fy

            norm = np.hypot(fx - sx, sy - fy)

            if norm == 0:
                continue

            cos = dot / norm
            if abs(cos) >= np.cos(self.cfg.max_theta_thresh):
                needed_lines.append(direct_line)

        return needed_lines

    def draw_staraight_line(self, binary: np.ndarray, straight_lines):
        straight_map = np.zeros_like(binary)

        for straight_line in straight_lines:
            for y, x in straight_line:
                straight_map[y, x] = 1

        return straight_map

    def vector_difference_theta(self, tan_vecs: np.ndarray):
        """
        the difference theta1 and theta2 as a vector difference
        """
        theta_diffs = []
        length = len(tan_vecs)
        for i in range(length - 1):
            vx1, vy1 = tan_vecs[i]
            vx2, vy2 = tan_vecs[(i + 1)]

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

    def horizontal_filter(
        self, binary: np.ndarray, cont_vecs: List, contours: List[np.ndarray]
    ):
        """
        judge a vector on a pixel is horizontal by comparing cos
        """
        filtered_map = np.zeros_like(binary)

        for i, tangent_vecs in enumerate(cont_vecs):
            for j, tangent_vec in enumerate(tangent_vecs):

                dotp = tangent_vec[0]  # dot production = vec[0]*1 + vec[1]*0 = vec[0]
                norm = np.hypot(tangent_vec[0], tangent_vec[1])

                if norm == 0:
                    continue

                cos = dotp / norm
                cos = np.clip(cos, -1.0, 1.0)

                if abs(cos) > np.cos(self.cfg.max_theta_thresh):
                    x, y = contours[i][j % len(contours[i])][0]
                    filtered_map[y, x] = 1

        return filtered_map

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
        - contour
        - tangent vector
        - detected horizontal line pixels
        """

        import matplotlib.pyplot as plt

        h, w = binary.shape

        # RGB化
        vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)

        # contour描画
        cv2.drawContours(
            vis,
            contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=1,
        )

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
        # 画像サイズに合わせてwindowを作成
        dpi = 100
        fig_w = vis.shape[1] / dpi
        fig_h = vis.shape[0] / dpi

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Contour + Tangent Vector + Horizontal Detection")
        plt.axis("off")

        # 余白除去
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.show()


if __name__ == "__main__":
    config = CNRConfig()

    segmenter = ContourNoiseRemover(config=config)
    segmenter.run(config.input_path)
