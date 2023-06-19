import cv2
import numpy as np


def generate_allzero_uint8_nparr(width: int, height: int):
    result = []
    for i_col in range(height):
        px_col = []
        for i_row in range(width):
            px_col.append([0, 0, 0])

        result.append(px_col)

    return np.array(result, dtype="uint8")


def detect_imdiff_by_saturation(
    img1_hsv: cv2.Mat,
    img2_hsv: cv2.Mat,
    TH_VALID_SATURATION_DIFF=5,
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


def detect_imdiff_by_bgrvalue(
    img1_bgr: cv2.Mat,
    img2_bgr: cv2.Mat,
    TH_VALID_BGRVALUE_MANHATTAN_DIFF=100,
    img_res=(1920, 1080),
):
    result_diff_img_bgr = generate_allzero_uint8_nparr(img_res[0], img_res[1])
    result_details = []

    for i_col in range(img1_bgr.__len__()):
        img1row = img1_bgr[i_col]
        img2row = img2_bgr[i_col]

        for i_row in range(img1row.__len__()):
            img1px = img1row[i_row]
            img2px = img2row[i_row]

            diff_d_manhattan = np.linalg.norm(img1px - img2px, ord=1)

            if diff_d_manhattan > TH_VALID_BGRVALUE_MANHATTAN_DIFF:
                # 差分が白だったら消す
                if img2px[0] < 240 and img2px[1] < 240 and img2px[2] < 240:
                    # Update result image
                    result_diff_img_bgr[i_col, i_row] = np.array(
                        img2px, dtype="uint8"
                    )

                    # Update reslut details
                    result_details.append(
                        [
                            i_row,  # position of x
                            i_col,  # position of y
                            diff_d_manhattan,  # size of diff
                            img1px,  # color mapping of x
                            img2px,
                        ]
                    )

    return np.array(result_diff_img_bgr, dtype="uint8"), result_details
