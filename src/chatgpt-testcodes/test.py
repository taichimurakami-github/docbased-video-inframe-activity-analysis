import cv2
import numpy as np
import os

# 1. 画像を２枚読み込む
data_base_dir = os.path.join(os.path.dirname(__file__), ".data")
image1_path = os.path.join(data_base_dir, "78.jpg")  # 1枚目の画像ファイルのパスを指定してください
image2_path = os.path.join(data_base_dir, "79.jpg")  # 2枚目の画像ファイルのパスを指定してください

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 2. 画素の差分を計算する
diff_image = cv2.absdiff(image1, image2)
cv2.imshow("result_absdiff", diff_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. 差分画像における差分がある画素の数をカウントする
diff_count = np.sum(diff_image > 30)
print("absdiff_diffcount", diff_count)

# 4. 差分画像における差分がある部分の輪郭を検出する
gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_diff, 1, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 5. 輪郭に基づいてマスク画像を作成する
mask = np.zeros_like(diff_image)
cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# 6. マスク画像を用いて画像を切り抜く
cropped_image = cv2.bitwise_and(image1, mask)

# 7. 切り抜いた画像を保存する
# output_path = "path/to/save/output.jpg"  # 出力画像ファイルのパスを指定してください
# cv2.imwrite(output_path, cropped_image)
cv2.imshow("result", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
