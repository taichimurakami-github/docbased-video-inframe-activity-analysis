import cv2
import numpy as np
from PIL import Image


class CV2ImageUtil:
    def load(self, file_path):
        return cv2.imread(file_path)

    def apply_bgr2gray(self, cv2img: cv2.Mat):
        return cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)

    def apply_gray2bgr(self, cv2img: cv2.Mat):
        return cv2.cvtColor(cv2img, cv2.COLOR_GRAY2BGR)

    def apply_bgr2hsv(self, cv2img: cv2.Mat):
        return cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)

    def apply_bgr2rgb(self, cv2img: cv2.Mat):
        return cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

    def apply_hsv2bgr(self, cv2img: cv2.Mat):
        return cv2.cvtColor(cv2img, cv2.COLOR_HSV2BGR)

    def apply_bgr2bin(self, cv2img: cv2.Mat):
        return cv2.threshold(
            self.apply_bgr2gray(cv2img), 0, 255, cv2.THRESH_OTSU
        )

    def apply_filter_gaussian_blur(
        self, cv2img: cv2.Mat, ksize=(13, 13), sigmaX=0
    ):
        return cv2.GaussianBlur(cv2img, ksize, sigmaX)

    def apply_clahe(
        self, cv2img: cv2.Mat, clipLimit=30.0, tileGridSize=(10, 10)
    ):
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        return clahe.apply(cv2img)

    # OpenCV.Image --> PIL.Image
    # グレースケール，rgbもしくはrgbaのいずれかのfmtのみOK．
    # 参考：https://qiita.com/derodero24/items/f22c22b22451609908ee
    def cvt_cv_to_pil(self, cv2img):
        new_image = cv2img.copy()

        if new_image.ndim == 2:  # モノクロ
            pass

        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)

        return Image.fromarray(new_image)

    # PIL.Image --> OpenCV.Image
    # 参考：https://qiita.com/derodero24/items/f22c22b22451609908ee
    def cvt_pil_to_cv(self, pilimg: Image):
        """PIL型 -> OpenCV型"""
        new_image = np.array(pilimg, dtype=np.uint8)

        if new_image.ndim == 2:  # モノクロ
            pass

        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)

        return new_image

    def get_absdiff(self, cv2img1: cv2.Mat, cv2img2: cv2.Mat):
        return cv2.absdiff(cv2img1, cv2img2)

    def show(self, cv2img: cv2.Mat, window_name: str):
        cv2.imshow(window_name, cv2img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, cv2img: cv2.Mat, file_path: str):
        cv2.imwrite(file_path, cv2img)
