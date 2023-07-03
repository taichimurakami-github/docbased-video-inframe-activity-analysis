import cv2
import numpy as np


def calculate_projection_profile_of_bgrimg(
    cv_bgr_img: cv2.Mat,
    th_img_binarization=100,
):
    cv_gray_img = cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2GRAY)
    _, cv_bin_img = cv2.threshold(
        cv_gray_img, th_img_binarization, 255, cv2.THRESH_BINARY
    )
    cv_bin_img[cv_bin_img == 0] = 1
    cv_bin_img[cv_bin_img == 255] = 0

    # img1に関する投影プロジェクションの計算
    horizontal_profile = np.sum(cv_bin_img, axis=1)
    vertical_profile = np.sum(cv_bin_img, axis=0)

    return (horizontal_profile, vertical_profile)


def calculate_projection_profile_of_binimg(
    cv_bin_img: cv2.Mat,
    black_px_value=0,
    white_px_value=255,
):
    cv_bin_img[cv_bin_img == black_px_value] = 1
    cv_bin_img[cv_bin_img == white_px_value] = 0

    # img1に関する投影プロジェクションの計算
    horizontal_profile = np.sum(cv_bin_img, axis=1)
    vertical_profile = np.sum(cv_bin_img, axis=0)

    return (horizontal_profile, vertical_profile)
