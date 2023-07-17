import cv2
import os
from analyzer.image.basics.im_convert import binarize_img
from analyzer.image.basics.im_testdata import get_data_dir
from analyzer.image.basics.im_filter import (
    perform_median_filter,
    perform_gaussian_filter,
)

# from analyzer.image.basics.im_histgram import plt_hsv_histgrams
# from analyzer.image.basics.im_bgcolor import (
#     replace_black_to_transparent,
#     detect_bgcolor_by_kmeans,
# )


def __crop_image_around_contour_as_square(
    imcrop_src: cv2.Mat,
    contours: cv2.Mat,
    original_dsize: tuple[int, int],
    resized_dsize: tuple[int, int],
    th_valid_area_size=1000,
):
    r_width = resized_dsize[0]
    r_height = resized_dsize[1]
    o_width = original_dsize[0]
    o_height = original_dsize[1]

    im_cropped = []
    im_cropped_bbox = []  # tlbr(top,left,bottom,right-ordered) fmtでbboxデータを格納

    # オリジナル画像の中から差分を切り取る
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # bboxの値を相対値として算出
        left = x / r_width
        top = y / r_height
        right = (x + w) / r_width
        bottom = (y + h) / r_height

        top_px = int(o_height * top)
        left_px = int(o_width * left)
        bottom_px = int(o_height * bottom)
        right_px = int(o_width * right)

        area_size = abs((top_px - bottom_px) * (left_px - right_px))

        if area_size > th_valid_area_size:
            # 元画像のサイズに変換したbboxの値を用いて，画像の切り抜きを実行
            imcrop_result = imcrop_src[
                top_px:bottom_px,  # y : y + h
                left_px:right_px,  # x : x + w
            ]
            im_cropped.append(imcrop_result)
            im_cropped_bbox.append(((top, left), (bottom, right)))

            cv2.imshow(f"imcropped_{i}", imcrop_result)

    return im_cropped, im_cropped_bbox


"""
=========NOTE: How to detect edges=========
ノイズ除去処理を行い，エッジを検出する
- https://www.jstage.jst.go.jp/article/jceeek/2020/0/2020_222/_pdf
- http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html
1. ガウシアンフィルタで平滑化（キャニー法のため，5x5サイズを適用）
2. キャニー法を用いたエッジ検出
3. モルフォロジー変換を用いてエッジの収縮処理
===========================================
"""


def extract_imdiff_by_absdiff(
    img1_original: cv2.Mat, img2_original: cv2.Mat, R_RESIZE=0.5
) -> tuple[list[cv2.Mat], list]:
    (
        original_height,
        original_width,
        orignal_channel,
    ) = img1_original.shape  # (height, width, channel)

    dsize = (int(original_width * R_RESIZE), int(original_height * R_RESIZE))
    print(f"dsize={dsize}")

    # ノイズの解像度低下及び処理負荷の軽減のため，リサイズを行う
    img1_resized = cv2.resize(img1_original, dsize)
    img2_resized = cv2.resize(img2_original, dsize)

    imdiff = cv2.absdiff(img1_resized, img2_resized)

    imblur_median = perform_median_filter(imdiff, 10)

    _, _, imcontours_base = binarize_img(imblur_median, 5)

    contours, _hierarchy = cv2.findContours(
        imcontours_base,
        # imblur_bilateral.copy(),
        # result_imabsdiff,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contours) == 0:
        print("No contours detected.")
        return [], []

    else:
        # 差分がある部分以外をブラックアウトした画像を作成
        img2_original_bitwised = cv2.bitwise_and(
            img2_original,
            cv2.cvtColor(
                cv2.resize(imcontours_base, (original_width, original_height)),
                cv2.COLOR_GRAY2BGR,
            ),
        )

        im_cropped, im_cropped_bbox = __crop_image_around_contour_as_square(
            imcrop_src=img2_original_bitwised,
            contours=contours,
            original_dsize=(original_width, original_height),
            resized_dsize=dsize,
        )

        imwithcontours = cv2.drawContours(
            image=imdiff.copy(),
            contours=contours,
            contourIdx=-1,
            color=(0, 0, 255),
            # color=255,
            thickness=-1,  # -1を指定すると塗りつぶしになる
            hierarchy=_hierarchy,
        )

        # cv2.imshow("imdiff", imdiff)
        cv2.imshow("imdiff_median_bin", imcontours_base)
        # cv2.imshow("imdiff_median_bin_upscaled", imcontours_base_upscaled)
        cv2.imshow("contours", imwithcontours)
        # cv2.imshow("im_bitwised", im_bitwised)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return im_cropped, im_cropped_bbox


if __name__ == "__main__":
    DATA_DIR = get_data_dir()

    test_datasets = [
        ("66", "67", "h-diff 01"),  # ハイライト差分01
        ("67", "110", "h-diff 02"),  # ハイライト差分02
        ("66", "110", "h-diff 03"),  # ハイライト差分03
        ("149", "150", "a-diff with h"),  # ハイライトあり，アノテーション差分
        ("52", "53", "same 01 no activities"),  # ハイライトなし，同じ
        ("131", "132", "same 02 highlighted"),  # ハイライトあり，同じ
    ]

    for dataset in test_datasets:
        img1_original = cv2.imread(os.path.join(DATA_DIR, f"{dataset[0]}.jpg"))
        img2_original = cv2.imread(os.path.join(DATA_DIR, f"{dataset[1]}.jpg"))

        print(f"\n\nAttempt: {dataset[2]}")
        extract_imdiff_by_absdiff(
            img1_original=img1_original, img2_original=img2_original
        )

    # PLT_XLIM = [0, 20]
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imdiff, cv2.COLOR_BGR2HSV),
    #     title="imdiff_abs",
    #     # plt_xlim=[0, 25],
    #     plt_xlim=PLT_XLIM,
    # )

    # cv2.imshow("imdiff_median", imblur_median)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_median, cv2.COLOR_BGR2HSV),
    #     title="immedian",
    # )

    # imblur_gaussian_1 = perform_gaussian_filter(imdiff, 1)
    # cv2.imshow("imblur_gaussian_1", imblur_gaussian_1)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_gaussian_1, cv2.COLOR_BGR2HSV),
    #     title="imblur_gaussian_1",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_gaussian_5 = perform_gaussian_filter(imdiff, 5)
    # cv2.imshow("imblur_gaussian_5", imblur_gaussian_5)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_gaussian_5, cv2.COLOR_BGR2HSV),
    #     title="imblur_gaussian_5",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_gaussian_10 = perform_gaussian_filter(imdiff, 10)
    # cv2.imshow("imblur_gaussian_10", imblur_gaussian_10)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_gaussian_10, cv2.COLOR_BGR2HSV),
    #     title="imblur_gaussian_10",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_gaussian_20 = perform_gaussian_filter(imdiff, 20)
    # cv2.imshow("imblur_gaussian_20", imblur_gaussian_20)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_gaussian_20, cv2.COLOR_BGR2HSV),
    #     title="imblur_gaussian_20",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_median_1 = perform_median_filter(imdiff, 1)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_median_1, cv2.COLOR_BGR2HSV),
    #     title="imblur_median_1",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_median_5 = perform_median_filter(imdiff, 5)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_median_5, cv2.COLOR_BGR2HSV),
    #     title="imblur_median_5",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_median_10 = perform_median_filter(imdiff, 10)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_median_10, cv2.COLOR_BGR2HSV),
    #     title="imblur_median_10",
    #     plt_xlim=PLT_XLIM,
    # )

    # imblur_median_20 = perform_median_filter(imdiff, 20)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imblur_median_10, cv2.COLOR_BGR2HSV),
    #     title="imblur_median_20",
    #     plt_xlim=PLT_XLIM,
    # )

    # imdiff_blur = perform_gaussian_filter(imdiff, n_perform=2)
    # _, imdiff_bin = cv2.threshold(imdiff_blur, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("imdiff_bin", imdiff)
