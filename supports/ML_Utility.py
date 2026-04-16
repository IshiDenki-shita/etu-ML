from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import cv2


class ConfigUtl:
    IMG_ROOT = Path("photos/")
    url_saito_cafe = "http://162.43.43.163:8080/api/v1/cafe"


class MLutility:
    def __init__(self, config):
        self.cfg = config or ConfigUtl()

    """
    File operation
    """

    def take_cell_imgs(self):
        imgs_root = self.cfg.IMG_ROOT
        dir_receives = sorted(list(imgs_root.iterdir()))[-1]

        img_dirs = [
            p
            for p in sorted(dir_receives.iterdir())
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        imgs = []
        for i, path in enumerate(img_dirs):
            img = cv2.imread(str(path))
            if img is None:
                print(f"{i + 1}番目の画像を取得できませんでした: {path}")
            imgs.append(img)

        print(f"{len(imgs)} imgs was taken")
        return imgs

    """
    Send JSON
    """

    def send_menu_json_to_saito(self, menus: list[tuple]) -> None:
        cafe_dict = self.format_mail_dict(menus=menus)
        res = requests.post(self.cfg.url_saito_cafe, json=cafe_dict)

        print("\nステータスコード\n")
        print(res.status_code)

        print("\nレスポンス本文\n")
        print(res.text)

    def format_mail_dict(self, menus: list[tuple]):
        formatted_menus = []
        for menu in menus:
            formatted_menus.append({"name": menu[0], "date": menu[1], "price": 500})
        mail = {
            "generated_at": self.now_jst_iso8601_seconds(),
            "menus": formatted_menus,
        }
        return mail

    # 斎藤VPSに送るJSONのgenerated_atの日付のフォーマットを固定する。
    def now_jst_iso8601_seconds(self) -> str:
        """
        斎藤VPS指定フォーマット:
        2024-12-02T14:30:00+09:00
        """
        JST = timezone(timedelta(hours=9))
        dt = datetime.now(JST).replace(microsecond=0)  # 秒までに丸める
        s = dt.isoformat()  # 'YYYY-MM-DDTHH:MM:SS+09:00'
        # 念のためオフセットが +0900 のようになったケースを +09:00 に補正
        if len(s) >= 5 and (s[-5] in ["+", "-"]) and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        return s
