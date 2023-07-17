import cv2
import os
from img_cvt_test import binarize_img
from get_data_dir import get_data_dir
from im_filter import perform_median_filter, perform_gaussian_filter
from im_histgram import plt_hsv_histgrams


if __name__ == "__main__":
    DATA_DIR = get_data_dir()
    print("\n\n")
    print(DATA_DIR)
    print("\n\n")

    img1 = cv2.imread(os.path.join(DATA_DIR, "149.jpg"))
    img2 = cv2.imread(os.path.join(DATA_DIR, "150.jpg"))

    R_RESIZE = 0.5

    h, w, c = img1.shape  # (height, width, channel)
    dsize = (int(w * R_RESIZE), int(h * R_RESIZE))
    print(f"dsize={dsize}")

    img1 = cv2.resize(img1, dsize)
    img2 = cv2.resize(img2, dsize)

    # img1_blur = perform_gaussian_filter(img1, n_perform=2)
    # img2_blur = perform_gaussian_filter(img2, n_perform=2)

    # _, img1_gray, img1_bin = binarize_img(img1_blur, 230)
    # _, img1_gray, img2_bin = binarize_img(img2_blur, 230)

    # cv2.imshow("img1_bin", img1_bin)
    # cv2.imshow("img2_bin", img2_bin)
    PLT_XLIM = [0, 20]

    imdiff = cv2.absdiff(img1, img2)
    cv2.imshow("imdiff", imdiff)
    # plt_hsv_histgrams(
    #     cv_img_hsv=cv2.cvtColor(imdiff, cv2.COLOR_BGR2HSV),
    #     title="imdiff_abs",
    #     # plt_xlim=[0, 25],
    #     plt_xlim=PLT_XLIM,
    # )

    # imdiff_gray = cv2.cvtColor(imdiff, cv2.COLOR_BGR2GRAY)
    imblur_median = perform_median_filter(imdiff, 10)
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

    imblur_gaussian_10 = perform_gaussian_filter(imdiff, 10)
    cv2.imshow("imblur_gaussian_10", imblur_gaussian_10)
    plt_hsv_histgrams(
        cv_img_hsv=cv2.cvtColor(imblur_gaussian_10, cv2.COLOR_BGR2HSV),
        title="imblur_gaussian_10",
        plt_xlim=PLT_XLIM,
    )

    imblur_gaussian_20 = perform_gaussian_filter(imdiff, 20)
    cv2.imshow("imblur_gaussian_20", imblur_gaussian_20)
    plt_hsv_histgrams(
        cv_img_hsv=cv2.cvtColor(imblur_gaussian_20, cv2.COLOR_BGR2HSV),
        title="imblur_gaussian_20",
        plt_xlim=PLT_XLIM,
    )

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

    # raise ValueError("Stop here.")

    # ノイズ除去処理を行い，エッジを検出する
    # - https://www.jstage.jst.go.jp/article/jceeek/2020/0/2020_222/_pdf
    # - http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html
    # 1. ガウシアンフィルタで平滑化（キャニー法のため，5x5サイズを適用）
    # 2. キャニー法を用いたエッジ検出
    # 3. モルフォロジー変換を用いてエッジの収縮処理

    # imblur_gaussian = perform_gaussian_filter(perform_gaussian_filter(imdiff))
    # cv2.imshow("gaussian_result", imblur_gaussian)

    # imblur_bilateral = perform_bilateral_filter(imblur_gaussian)
    # cv2.imshow("bilateral_result", imblur_bilateral)

    imblur_median_hsv = cv2.cvtColor(imblur_median, cv2.COLOR_BGR2HSV)

    _, _, imcontours_base = binarize_img(imblur_median, 10)
    cv2.imshow("imdiff_median_bin", imcontours_base)

    contours, _hierarchy = cv2.findContours(
        imcontours_base,
        # imblur_bilateral.copy(),
        # result_imabsdiff,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
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
    cv2.imshow("contours", imwithcontours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
