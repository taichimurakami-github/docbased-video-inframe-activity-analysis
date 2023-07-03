import cv2
import os
from get_data_dir import get_data_dir


def binarize_img(cv_img_bgr: cv2.Mat, bin_threshold=100):
    cv_img_gray = cv2.cvtColor(cv_img_bgr.copy(), cv2.COLOR_BGR2GRAY)

    _, cv_img_bin = cv2.threshold(
        cv_img_gray, bin_threshold, 255, cv2.THRESH_BINARY
    )

    # cv2.imshow("original", cv_img_bgr)
    # cv2.imshow(
    #     "grayscaled", cv2.cvtColor(cv_img_gray.copy(), cv2.COLOR_GRAY2BGR)
    # )
    # cv2.imshow(
    #     "binarized", cv2.cvtColor(cv_img_bin.copy(), cv2.COLOR_GRAY2BGR)
    # )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (cv_img_bgr, cv_img_gray, cv_img_bin)


if __name__ == "__main__":
    SRC_DIRPATH = get_data_dir()

    # img_src = os.path.join(SRC_DIRPATH, "67.jpg")
    img_src = os.path.join(SRC_DIRPATH, "66.jpg")

    cv_img_bgr = cv2.imread(img_src)

    binarize_img(cv_img_bgr, 235)