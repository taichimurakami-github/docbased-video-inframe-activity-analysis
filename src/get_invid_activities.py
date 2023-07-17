import os
import json
from sklearn.cluster import KMeans

import cv2
import numpy as np

from utils.CV2VideoUtil import CV2VideoUtil
from analyzer.image.basics.im_crop import crop_around_contour_as_square
from analyzer.image.basics.im_bgcolor import (
    replace_black_to_transparent,
)
from analyzer.image.detect_imdiff_by_saturation import (
    detect_imdiff_by_saturation,
)
from analyzer.image.detect_imdiff_by_bgrvalue import count_diff_pixel
from analyzer.image.basics.im_filter import perform_median_filter
from analyzer.image.basics.im_convert import binarize_img

DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), ".data")
VIDEO_FILE_PATH = os.path.join(DATA_BASE_DIR, "EdanMeyerVpt.mp4")


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


def get_invid_activities_as_img(
    img1src,
    img2src,
    OUTPUT_DIR: str | None = None,
    viewport_bbox: tuple[(int, int), (int, int)] = ((0, 0), (0, 0)),
    document_bbox: tuple[(int, int), (int, int)] = ((0, 0), (0, 0)),
    imshow=True,
):
    # image = CV2ImageUtil()
    img1_bgr = cv2.resize(cv2.imread(img1src), (960, 540))
    img2_bgr = cv2.resize(cv2.imread(img2src), (960, 540))
    # img1_bgr = cv2.imread(img1src)
    # img2_bgr = cv2.imread(img2src)
    # print(count_diff_pixel(img1_bgr, img2_bgr))
    # return
    # img2_bgr_original = cv2.resize()
    # bgcolor_rgb = get_bgcolor_by_kmeans(img1_bgr)

    img1_hsv = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2HSV)

    # img1_hsv = cv2.cvtColor(
    #     cv2.resize(img1_bgr.copy(), (1920 / 4, 1080 / 4)), cv2.COLOR_BGR2HSV
    # )
    # img2_hsv = cv2.cvtColor(
    #     cv2.resize(img2_bgr.copy(), (1920 / 4, 1080 / 4)), cv2.COLOR_BGR2HSV
    # )

    # 1. 単純な画素差分検出
    # result_imabsdiff = cv2.absdiff(img1_bgr, img2_bgr)
    # cv2.imshow("absdiff_example", result_imabsdiff)

    # 2. bgrベースの差分検出(with nose filter)
    # (result_diff_img_bgr, result_details) = detect_imdiff_by_bgrvalue(
    #     img1_bgr, img2_bgr
    # )
    # image.show(result_diff_img_bgr, "result_bgrpxbased_example")

    # 3. hsvベースの差分検出
    (result_diff_img_hsv, result_details) = detect_imdiff_by_saturation(
        img1_hsv, img2_hsv, TH_VALID_SATURATION_DIFF=1
    )

    # print(
    #     f"img len : {result_diff_img_hsv.__len__()} x {result_diff_img_hsv[0].__len__()}"
    # )
    # print(f"details len : {result_details.__len__()}")

    imabsdiff_gray = cv2.cvtColor(
        result_diff_img_hsv.copy(), cv2.COLOR_BGR2GRAY
    )
    # print(imabsdiff_gray)
    # _, imabsdiff_gray = cv2.threshold(
    #     imabsdiff_gray, 1, 255, cv2.THRESH_BINARY
    # )
    # print(imabsdiff_gray)
    # imabsdiff_gray = cv2.GaussianBlur(imabsdiff_gray, (5, 5), 0)

    contours, _hierarchy = cv2.findContours(
        imabsdiff_gray.copy(),
        cv2.RETR_EXTERNAL,
        # cv2.RETR_TREE,
        # cv2.RETR_LIST,
        # cv2.CHAIN_APPROX_NONE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # imcontours = cv2.drawContours(
    #     # image=imabsdiff_gray.copy(),
    #     image=cv2.cvtColor(result_diff_img_hsv.copy(), cv2.COLOR_HSV2BGR),
    #     contours=contours,
    #     contourIdx=-1,
    #     color=(0, 0, 255),
    #     # color=255,
    #     # thickness=-1,  # -1を指定すると塗りつぶしになる
    #     thickness=2,  # -1を指定すると塗りつぶしになる
    #     hierarchy=_hierarchy,
    # )

    imcontours = cv2.cvtColor(result_diff_img_hsv.copy(), cv2.COLOR_HSV2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        imcontours = cv2.rectangle(
            imcontours,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    print(img2_bgr.shape)
    print(imabsdiff_gray.shape)
    im_bitwise_result = cv2.bitwise_and(
        img2_bgr.copy(), cv2.cvtColor(imabsdiff_gray, cv2.COLOR_GRAY2BGR)
    )

    # 切り抜いた画像を格納するリスト
    cropped_images, cropped_immeta = crop_around_contour_as_square(
        im_bitwise_result, contours
    )

    # 5. 切り抜いた画像を保存する
    print(cropped_images.__len__())
    vp_imheight, vp_imwidth, _ = img2_bgr.shape
    print(f"base image shape = ({vp_imwidth, vp_imheight}) ")

    if OUTPUT_DIR != None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_immeta = {
        "base_immeta": [
            (vp_imwidth, vp_imheight),  # base image(viewport) shape
            viewport_bbox,  # base image bbox(px)
            document_bbox,  # base document bbox(px)
        ],
        "cropped_immeta": [],
    }

    result_imgs = []

    for i, cropped_image in enumerate(cropped_images):
        # cv2.imshow(f"Cropped Image {i+1}", cropped_image)
        # print(cv2.boundingRect(cropped_immeta[i]))
        x, y, w, h = cropped_immeta[i]
        bbox = (
            (x / vp_imwidth, y / vp_imheight),
            ((x + w) / vp_imwidth, (y + h) / vp_imheight),
        )
        print(f"\nCropped Image {i} : (x,y,w,h) = {cropped_immeta[i]})")
        print(f"realtive_bbox = {bbox}")

        result_immeta["cropped_immeta"].append((bbox, (x, y, w, h)))

        # convert 3channel image to 4channel image
        # by converting black pixels to transparent pixels
        cropped_image_with_alpha = replace_black_to_transparent(
            cropped_image, (0, 0, 0)
        )
        result_imgs.append(cropped_image_with_alpha)

        if imshow == True:
            cv2.imshow(
                f"Cropped Image with alpha: {i}", cropped_image_with_alpha
            )

        if OUTPUT_DIR != None:
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"{i}.png"),
                cropped_image_with_alpha,
            )

    if OUTPUT_DIR != None:
        with open(os.path.join(OUTPUT_DIR, "immeta.json"), "w") as fp:
            json.dump(fp=fp, obj=result_immeta)

    if imshow == True:
        cv2.imshow(
            "result_saturationbased_example",
            cv2.cvtColor(result_diff_img_hsv, cv2.COLOR_HSV2RGB),
        )
        cv2.imshow("draw_coutours_result", imcontours)
        cv2.imshow("result_bitwise_and", im_bitwise_result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_imgs, result_immeta


get_invid_activities_as_img(
    # os.path.join(DATA_BASE_DIR, "52.jpg"),
    # os.path.join(DATA_BASE_DIR, "53.jpg"),
    # os.path.join(DATA_BASE_DIR, "66.jpg"),
    # os.path.join(DATA_BASE_DIR, "67.jpg"),
    # os.path.join(DATA_BASE_DIR, "77.jpg"),
    # os.path.join(DATA_BASE_DIR, "78.jpg"),
    # os.path.join(DATA_BASE_DIR, "106.jpg"),
    # os.path.join(DATA_BASE_DIR, "107.jpg"),
    # os.path.join(DATA_BASE_DIR, "108.jpg"),
    # os.path.join(DATA_BASE_DIR, "109.jpg"),
    os.path.join(DATA_BASE_DIR, "110.jpg"),
    os.path.join(DATA_BASE_DIR, "119.jpg"),
    # os.path.join(DATA_BASE_DIR, "131.jpg"),
    # os.path.join(DATA_BASE_DIR, "132.jpg"),
    # os.path.join(DATA_BASE_DIR, "149.jpg"),
    # os.path.join(DATA_BASE_DIR, "150.jpg"),
    # os.path.join(DATA_BASE_DIR, "307.jpg"),
    # os.path.join(DATA_BASE_DIR, "308.jpg"),
    # os.path.join(DATA_BASE_DIR, "309.jpg"),
    # OUTPUT_DIR=os.path.join(
    #     DATA_BASE_DIR, "cropped_images", datetime.now().timestamp().__str__()
    # ),
    imshow=True,
)
