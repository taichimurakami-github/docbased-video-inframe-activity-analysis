# coding: utf-8
from tkinter import Image
import numpy as np
import copy
import cv2


def pic_clipping(path_list, pram_list, transparent):
    print("pic_clipping")
    # 画像の読み込み
    img = cv2.imread(path_list[0])
    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 白黒二極化
    ret, thresh = cv2.threshold(
        gray, int(pram_list[0]), 255, cv2.THRESH_BINARY
    )
    # 輪郭の抽出
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    maxsize = int(pram_list[1])
    minsize = int(pram_list[2])
    # スラッシュの置換
    dst_path = path_list[1].replace("/", "\\\\")

    # 元画像のコピー
    output = copy.copy(img)
    for i in range(len(contours)):
        cnt = contours[i]
        # 誤検出と思われる大きすぎるサイズの排除、ノイズと思われる小さいサイズの排除
        if cv2.contourArea(cnt) > maxsize or cv2.contourArea(cnt) < minsize:
            continue
        # 元画像に検出部分をマークする
        output = cv2.drawContours(output, [cnt], 0, (0, 255, 0), -1)
        # 輪郭からバンディングボックスを計算
        rect = cv2.boundingRect(cnt)
        # 検出対象を矩形でトリミングする
        crop_img = img[
            rect[1] : rect[1] + rect[3] - 1, rect[0] : rect[0] + rect[2] - 1
        ]

        # 白地の透過処理を行うかどうか
        if transparent:
            dst = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
            # dst[:,:, 3] = np.where(np.all(dst == 255, axis=-1), 0, 255)
            # 透過色の下限値
            color_lower = np.array([200, 200, 200, 255])
            # 透過色の上限値
            color_upper = np.array([255, 255, 255, 255])
            img_mask = cv2.inRange(dst, color_lower, color_upper)
            img_bool = cv2.bitwise_not(dst, dst, mask=img_mask)
            cv2.imwrite(dst_path + "\\picture" + str(i) + ".png", img_bool)
        else:
            cv2.imwrite(dst_path + "\\picture" + str(i) + ".png", crop_img)
    cv2.imwrite(dst_path + "\\detection_parts.png", output)
