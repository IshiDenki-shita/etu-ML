"""
purpose: extraction characters's binary img from sentence img
author: Mats

info: I uploaded a doc explaining about how deal with charless area(appear later)
      on our sharing drive
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# to find original package
sys.path.append(str(Path(__file__).resolve().parent.parent))

# original classes
from supports.OHRSM_CNN import menu_classifier, HiraganaCNN
from supports.MTMT_CharSeg import CharacterSegmenter
from supports.ML_Utility import MLutility


@dataclass
class ConfigSeg:
    MIN_CHAR_WIDTH: int = 60  # for char detection
    CHAR_WIDTH_RATIO: float = 0.05
    SMOOTH_KERNEL: int = 5
    MAX_BLANK_DENSITY: float = 0.03
    MIN_CHAR_DENSITY: float = 0.08
    ARROW_WID_RATIO: float = 1 / 30
    ARROW_AVE_SURFACE: int = 450
    THIN_NOISE_WIDTH: int = 3  # for partial blank detection
    UPPER_BLANK_RATIO: float = 0.5
    LEFT_BLANK_RATIO: float = 0.05
    RIGHT_BLANK_RATIO: float = 0.5
    NOISE_HEIGHT: int = 5
    IGNORE_WID: int = 10
    PHOTO_HW: tuple[int, int] = (64, 64)  # for regulate sizes of photos


class ConfigUtl:
    IMG_ROOT = Path("photos/")
    url_saito_cafe = "http://162.43.43.163:8080/api/v1/cafe"


# main function
if __name__ == "__main__":
    print(Path("photos/").resolve())
    seg = CharacterSegmenter(config=ConfigSeg())
    cnn = menu_classifier()
    utl = MLutility(config=ConfigUtl())

    menus = []
    cell_imgs = utl.take_cell_imgs()

    for i, img in enumerate(cell_imgs):
        print(i, type(img), img is None)

    if not cell_imgs:
        print("画像を取得できませんでした。")
        sys.exit()

    for i, img in enumerate(cell_imgs):
        print(img)
        print(f"{i + 1}番目のセル")
        cell = seg.run(img=img)
        # seg.visualize(img, binary, proj, areas)

        name = cnn.predict_sentence(cell=cell)

        if name == 1 and len(menus) >= 1:
            menus.append(menus[i - 1])
        elif name == 0:
            continue

        menus.append(name)
        print(name, end="\n\n")

    # utl.send_menu_json_to_saito(menus=menus)
    # making JSON
    # sending to SaitoVPS
