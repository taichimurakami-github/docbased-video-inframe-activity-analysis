import cv2
import os
from datetime import datetime
from basics.im_testdata import get_data_dir
from basics.im_convert import binarize_img
from basics.im_contour import get_area_of_contours


def show_with_contours(img_bin_src: str, WRITE_DIR: str):
    imgname = os.path.basename(img_bin_src).split(".")[0]
    img_bin = cv2.imread(img_bin_src, cv2.IMREAD_GRAYSCALE)
    contours, _hierarchy = cv2.findContours(
        img_bin.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    print(f"\nArea of contours result: {imgname}")
    print(get_area_of_contours(contours))

    img_with_contours = cv2.drawContours(
        image=cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR),
        contours=contours,
        contourIdx=-1,
        color=(0, 0, 255),
        # color=255,
        thickness=3,  # -1を指定すると塗りつぶしになる
        hierarchy=_hierarchy,
    )

    cv2.imshow(imgname, img_with_contours)
    cv2.imwrite(os.path.join(WRITE_DIR, f"{imgname}.jpg"), img_with_contours)


if __name__ == "__main__":
    DATA_DIR = get_data_dir()
    frames = ["122-123", "173-174", "195-196", "235-236", "278-279"]

    WRITE_DIR = os.path.join(
        DATA_DIR, "contour_test", datetime.now().strftime("%y-%m-%d_%H%M%S")
    )
    print(f"{WRITE_DIR}\n")

    os.makedirs(WRITE_DIR, exist_ok=True)

    for f in frames:
        show_with_contours(os.path.join(DATA_DIR, f"frame_{f}.png"), WRITE_DIR)

    # cv2.waitKey(0)
