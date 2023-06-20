import cv2
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
        return cv2.threshold(cv2img, 0, 255, cv2.THRESH_OTSU)

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
    def cvt_to_pil_img(self, cv2img_rgb):
      return Image.fromarray(cv2img_rgb.copy())

    def get_absdiff(self, cv2img1: cv2.Mat, cv2img2: cv2.Mat):
        return cv2.absdiff(cv2img1, cv2img2)

    def show(self, cv2img: cv2.Mat, window_name: str):
        cv2.imshow(window_name, cv2img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, cv2img: cv2.Mat, file_path: str):
        cv2.imwrite(file_path, cv2img)
