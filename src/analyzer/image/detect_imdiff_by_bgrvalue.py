import cv2
import numpy as np
from .basics.generate_allzero_uint8_nparr import generate_allzero_uint8_nparr
from .detect_imdiff_by_saturation import detect_imdiff_by_saturation


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


def count_diff_pixel(img1_bgr, img2_bgr):
    im_absdiff = cv2.absdiff(img1_bgr.copy(), img2_bgr.copy())
    cv2.imshow("abs", im_absdiff)
    img, r = detect_imdiff_by_saturation(
        cv2.cvtColor(img1_bgr.copy(), cv2.COLOR_BGR2HSV),
        cv2.cvtColor(img2_bgr.copy(), cv2.COLOR_BGR2HSV),
    )
    print(r)
    _, im_absdiff_bin = cv2.threshold(
        cv2.cvtColor(im_absdiff, cv2.COLOR_BGR2GRAY),
        30,
        255,
        cv2.THRESH_BINARY,
    )

    cv2.imshow("test", im_absdiff_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.sum(im_absdiff_bin > 0)
