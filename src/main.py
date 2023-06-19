import os
import numpy as np
import json
from pprint import pprint
from sklearn.cluster import KMeans
import cv2
from PIL import Image
from utils.CV2VideoUtil import CV2VideoUtil
from utils.CV2ImageUtil import CV2ImageUtil
from analyzer.image import (
    detect_imdiff_by_bgrvalue,
    detect_imdiff_by_saturation,
    generate_allzero_uint8_nparr,
)

data_base_dir = os.path.join(os.path.dirname(__file__), ".data")
video_file_path = os.path.join(data_base_dir, "EdanMeyerVpt.mp4")


def get_video_frame_every_sec_from_src(src: str):
    video = CV2VideoUtil()
    cap = video.load(src)
    _, result_img = video.get_video_frame_every_sec(cap, 6)
    return result_img


def detect_and_draw_contours_by_kmeans(
    img_background: cv2.Mat, img_filtered: cv2.Mat
):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    img_fgmask = fgbg.apply(img_background)
    img_fgmask = fgbg.apply(img_filtered)

    # 検出した矩形領域でクリッピング(クラスタリングを行い，その結果からオブジェクト領域を検出する)
    N_CLUSTERS = 6
    Y, X = np.where(img_fgmask > 200)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit_predict(
        np.array([X, Y]).T
    )

    def get_x_y_limit(Y, X, result, cluster):
        NO = np.where(result == cluster)
        x_max = np.max(X[NO])
        x_min = np.min(X[NO])
        y_max = np.max(Y[NO])
        y_min = np.min(Y[NO])

        x_max = int(x_max)
        x_min = int(x_min)
        y_max = int(y_max)
        y_min = int(y_min)
        return x_min, y_min, x_max, y_max

    def bounding_box(img, x_min, y_min, x_max, y_max):
        img = cv2.rectangle(
            img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5
        )
        return img

    # bounding boxを描画
    for i in range(0, N_CLUSTERS):
        x_min, y_min, x_max, y_max = get_x_y_limit(Y, X, kmeans, i)
        img_filtered = bounding_box(img_filtered, x_min, y_min, x_max, y_max)

    return img_filtered


def compare_2_imgs(img1src, img2src):
    image = CV2ImageUtil()

    # imgの構造
    # [ <- len = 1080(num of pixels of column)
    #  [
    #    [255,255,255] <- depends on colorspace (ex: [b,g,r], [h,s,v])
    #    [255,255,255]
    #    ...
    #    [255,255,255]
    #  ], <- len = 1920(num of pixels of row)
    #  [
    #    [255,255,255]
    #    [255,255,255]
    #    ...
    #    [255,255,255]
    #  ], <- len = 1920(num of pixels of row)
    #  ...
    #  [
    #    [255,255,255]
    #    [255,255,255]
    #    ...
    #    [255,255,255]
    #  ], <- len = 1920(num of pixels of row)
    # ]

    img1_bgr = image.load(img1src)
    img2_bgr = image.load(img2src)

    img1_hsv = image.apply_bgr2hsv(img1_bgr)
    img2_hsv = image.apply_bgr2hsv(img2_bgr)

    # img1 = image.apply__filter_gaussian_blur(image.apply_clahe(image.apply_bgr2gray(img1)))
    # img2 = image.apply__filter_gaussian_blur(image.apply_clahe(image.apply_bgr2gray(img2)))

    # 1. 単純な画素差分検出
    # result = image.get_absdiff(img1_bgr, img2_bgr)
    # image.show(result, "absdiff_example")

    # 2. bgrベースの差分検出(with nose filter)
    # (result_diff_img_bgr, result_details) = detect_imdiff_by_bgrvalue(
    #     img1_bgr, img2_bgr
    # )
    # image.show(result_diff_img_bgr, "result_bgrpxbased_example")

    # 3. hsvベースの差分検出
    (result_diff_img_hsv, result_details) = detect_imdiff_by_saturation(
        img1_hsv, img2_hsv
    )
    # image.show(
    #     image.apply_hsv2bgr(result_diff_img_hsv),
    #     "result_saturationbased_example",
    # )
    print(
        f"img len : {result_diff_img_hsv.__len__()} x {result_diff_img_hsv[0].__len__()}"
    )
    print(f"details len : {result_details.__len__()}")

    # 切り出した画像のうち，矩形領域でない部分を検出
    # img_background = generate_allzero_uint8_nparr(1920, 1080)
    # img_filtered = image.apply_filter_gaussian_blur(
    #     image.apply_hsv2bgr(result_diff_img_hsv), (5, 5), 0
    # )
    # image.show(img_filtered, "cropping_test_result")

    tmp_path = os.path.join(os.path.dirname(__file__), ".data", "_temp.png")
    image.save(
        image.apply_hsv2bgr(result_diff_img_hsv),
        tmp_path,
    )

    # return

    img_base = image.apply_bgr2gray(
        image.apply_filter_gaussian_blur(image.load(tmp_path), (5, 5), 0)
    )

    contours, hierarchy = cv2.findContours(
        img_base,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    img_disp = image.apply_gray2bgr(img_base)

    cv2.drawContours(
        image=img_disp,
        contours=contours,
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=4,  # -1を指定すると塗りつぶしになる
        hierarchy=hierarchy,
    )

    # for contour in contours:
    #     for point in contour:
    #         cv2.circle(img_disp, point[0], 3, (0, 255, 0), -1)

    image.show(img_disp, "cropping_test_result")


# image = CV2ImageUtil()
# j = 0
# for i in range(len(result_img) - 1):
#   # prev_frame_gray = image.apply_grayscale(result_img[i])
#   # current_frame_gray = image.apply_grayscale(result_img[i+1])

#   image.save(result_img[i], os.path.join(data_base_dir, f"frame_"))

#   j = j + 1
#   if j > 3:
#     return


compare_2_imgs(
    # os.path.join(data_base_dir, "66.jpg"),
    # os.path.join(data_base_dir, "67.jpg"),
    os.path.join(data_base_dir, "149.jpg"),
    os.path.join(data_base_dir, "150.jpg"),
)
