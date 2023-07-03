import cv2
import numpy as np
from basics.generate_allzero_uint8_nparr import generate_allzero_uint8_nparr


def detect_imdiff_by_saturation(
    img1_hsv: cv2.Mat,
    img2_hsv: cv2.Mat,
    TH_VALID_SATURATION_DIFF=10,
    img_res=(1920, 1080),
):
    result_diff_img_hsv = generate_allzero_uint8_nparr(img_res[0], img_res[1])
    result_details = []

    for i_col in range(img1_hsv.__len__()):
        img1row = img1_hsv[i_col]
        img2row = img2_hsv[i_col]

        for i_row in range(img1row.__len__()):
            img1px_s = img1row[i_row][1]
            img2px_s = img2row[i_row][1]

            # 変化前後のpxの彩度差が一定以下の場合はノイズとして除去
            saturation_diff_abs = abs(int(img1px_s) - int(img2px_s))
            if saturation_diff_abs < TH_VALID_SATURATION_DIFF:
                continue

            # Update result image
            result_diff_img_hsv[i_col, i_row] = np.array(
                img2row[i_row], dtype="uint8"
            )

            # Update reslut details
            result_details.append(
                [
                    (
                        i_row,
                        i_col,
                        saturation_diff_abs,
                    ),  # position of detected diff pixel
                    img1row[i_row],
                    img2row[i_row],
                ]
            )

    return np.array(result_diff_img_hsv, dtype="uint8"), result_details
