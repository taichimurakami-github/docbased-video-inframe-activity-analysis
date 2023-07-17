import cv2
import numpy as np


def detect_bgcolor_by_kmeans(
    cv_img_bgr: cv2.Mat,
) -> tuple[tuple[int, int, int], str]:
    # 1. 画像をカラーで読み込む
    # image = cv2.imread("image.jpg")

    # 2. 読み込んだ画像について、背景色を検出する
    # 画像の最頻値を求める
    pixels = np.float32(cv_img_bgr.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 最頻値を背景色として使用する
    background_color = np.uint8(palette[0])

    # 3. 検出した背景色のカラーコードを表示する
    bg_color_code = tuple(int(c) for c in background_color)
    bg_color_code_hex = f"#{hex(bg_color_code[2])[2:]}{hex(bg_color_code[1])[2:]}{hex(bg_color_code[0])[2:]}"
    print("pallete:", palette)
    print("Background Color Code:", bg_color_code, bg_color_code_hex)

    return bg_color_code, bg_color_code_hex


def replace_black_to_transparent(image: cv2.Mat, pixel=(0, 0, 0)) -> cv2.Mat:
    """
    # https://stackoverflow.com/questions/70223829/opencv-how-to-convert-all-black-pixels-to-transparent-and-save-it-to-png-file
    import cv2
    import numpy as np

    # load image
    img = cv2.imread('girl_on_black.png')

    # threshold on black to make a mask
    color = (0,0,0)
    mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite('girl_on_black_transparent.png', result)

    # display result, though it won't show transparency
    cv2.imshow("MASK", mask)
    cv2.imshow("RESULT", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    tmp_image = image.copy()

    mask = np.where((tmp_image == pixel).all(axis=2), 0, 255).astype(np.uint8)

    imresult = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2BGRA)
    imresult[:, :, 3] = mask

    return imresult
