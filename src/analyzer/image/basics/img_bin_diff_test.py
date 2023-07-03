import cv2
import os
from img_cvt_test import binarize_img
from get_data_dir import get_data_dir
from plot_edges import plot_cv_img_bin_with_edges


def perform_gaussian_filter(imdiff, n_perform=1, kernel=(5, 5)):
    n = n_perform
    imresult = imdiff.copy()

    while n > 0:
        imresult = cv2.GaussianBlur(imresult, kernel, 0)
        n -= 1

    return imresult


def perform_bilateral_filter(imdiff, n_perform=1):
    n = n_perform
    imresult = imdiff.copy()

    while n > 0:
        imresult = cv2.bilateralFilter(imresult, 9, 75, 75)
        n -= 1

    return imresult


if __name__ == "__main__":
    DATA_DIR = get_data_dir()

    img1 = cv2.imread(os.path.join(DATA_DIR, "66.jpg"))
    img2 = cv2.imread(os.path.join(DATA_DIR, "67.jpg"))

    img1_blur = perform_gaussian_filter(img1, n_perform=2)
    img2_blur = perform_gaussian_filter(img2, n_perform=2)

    _, _, img1_bin = binarize_img(img1_blur, 230)
    _, _, img2_bin = binarize_img(img2_blur, 230)

    imdiff = cv2.absdiff(img1_bin, img2_bin)

    cv2.imshow("imdiff", imdiff)

    imdiff_blur = perform_gaussian_filter(imdiff, n_perform=2)
    _, imdiff_bin = cv2.threshold(imdiff_blur, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("imdiff_bin", imdiff)

    # raise ValueError("Stop here.")

    # ノイズ除去処理を行い，エッジを検出する
    # - https://www.jstage.jst.go.jp/article/jceeek/2020/0/2020_222/_pdf
    # - http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html
    # 1. ガウシアンフィルタで平滑化（キャニー法のため，5x5サイズを適用）
    # 2. キャニー法を用いたエッジ検出
    # 3. モルフォロジー変換を用いてエッジの収縮処理

    imblur_gaussian = perform_gaussian_filter(perform_gaussian_filter(imdiff))
    cv2.imshow("gaussian_result", imblur_gaussian)

    imblur_bilateral = perform_bilateral_filter(imblur_gaussian)
    cv2.imshow("bilateral_result", imblur_bilateral)

    contours, _hierarchy = cv2.findContours(
        imblur_bilateral.copy(),
        # result_imabsdiff,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    imwithcontours = cv2.drawContours(
        image=imblur_bilateral.copy(),
        contours=contours,
        contourIdx=-1,
        # color=(0, 0, 255),
        color=255,
        thickness=-1,  # -1を指定すると塗りつぶしになる
        hierarchy=_hierarchy,
    )
    cv2.imshow("contours", imwithcontours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
