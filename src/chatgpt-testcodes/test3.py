import os
import cv2
import numpy as np

# 1. ２枚の画像をカラーで読み込む
data_base_dir = os.path.join(os.path.dirname(__file__), ".data")
image1_path = os.path.join(data_base_dir, "66.jpg")  # 1枚目の画像ファイルのパスを指定してください
image2_path = os.path.join(data_base_dir, "67.jpg")  # 2枚目の画像ファイルのパスを指定してください

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 2. ２枚の画像に対し、cv2.absdiff()を適用して計算する
diff = cv2.absdiff(image1, image2)

# 3. 計算結果から、cv2.findContours()を使用して輪郭を見つける
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(
    gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 4. cv2.drawContours()を用いて輪郭を描画する
result = np.copy(image1)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# 結果の画像を表示
cv2.imshow("Contours", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
