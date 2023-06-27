import os
import cv2
import numpy as np

# 1. 2枚のカラー画像を読み込む
data_base_dir = os.path.join(os.path.dirname(__file__), ".data")
image1_path = os.path.join(data_base_dir, "66.jpg")  # 1枚目の画像ファイルのパスを指定してください
image2_path = os.path.join(data_base_dir, "68.jpg")  # 2枚目の画像ファイルのパスを指定してください

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 2. 視覚的に認められるレベルの画素の差分を検出する
diff_image = cv2.absdiff(image1, image2)

# 3. ノイズフィルタを適用して差分画像のノイズを除去する
filtered_diff = cv2.GaussianBlur(diff_image, (5, 5), 0)

hsv_filtered_diff = cv2.cvtColor(filtered_diff, cv2.COLOR_BGR2HSV)
lower_saturation = 50
for row in hsv_filtered_diff:
    for px in row:
        if px[1] < lower_saturation:
            px[1] = 0

filtered_diff = cv2.cvtColor(hsv_filtered_diff, cv2.COLOR_HSV2BGR)

px = np.sum(filtered_diff > 0)
print(px)

cv2.imshow("filtered_imdiff", filtered_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. 差分がある領域をピンポイントで切り抜いて画像を作成する
gray_diff = cv2.cvtColor(filtered_diff, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_diff, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 切り抜いた画像を格納するリスト
cropped_images = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image1[y : y + h, x : x + w]  # 画像1から切り抜く
    cropped_images.append(cropped_image)

# 5. 切り抜いた画像を表示する
for i, cropped_image in enumerate(cropped_images):
    cv2.imshow(f"Cropped Image {i+1}", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
