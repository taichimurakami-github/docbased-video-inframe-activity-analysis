import cv2
import numpy as np


def get_bgcolor_by_kmeans(cv_img_bgr):
    # 1. 画像をカラーで読み込む
    # image = cv2.imread("image.jpg")

    # 2. 読み込んだ画像について、背景色を検出する
    # 画像の最頻値を求める
    pixels = np.float32(cv_img_bgr.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 最頻値を背景色として使用する
    background_color = np.uint8(palette[0])

    # 3. 検出した背景色のカラーコードを表示する
    background_color_code = tuple(int(c) for c in background_color)
    print("Background Color Code:", background_color_code)
