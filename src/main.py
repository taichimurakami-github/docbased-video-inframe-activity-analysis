import os
import numpy as np
from pprint import pprint
import cv2
from PIL import Image
from utils.CV2VideoUtil import CV2VideoUtil
from utils.CV2ImageUtil import CV2ImageUtil

data_base_dir = os.path.join(os.path.dirname(__file__), ".data")
video_file_path = os.path.join(data_base_dir, "EdanMeyerVpt.mp4")


def get_video_frame_every_sec_from_src(src: str):
  video = CV2VideoUtil()
  cap = video.load(src)
  _, result_img =  video.get_video_frame_every_sec(cap, 6)
  return result_img

def generate_allzero_uint8_nparr(width: int, height: int):
  result = []
  for i_col in range(height):
    px_col = []
    for i_row in range(width):
      px_col.append([0,0,0])

    result.append(px_col)

  return np.array(result, dtype="uint8")


def compare_2_imgs(img1src, img2src):
  image = CV2ImageUtil()

  # imgの構造
  # [ <- len = 1080(num of pixels of column)
  #  [
  #    [255,255,255]
  #    [255,255,255]
  #    ...
  #    [255,255,255]
  #  ], <- len = 1920(num of pixels of row)
  #  [
  #    [255,255,255]
  #    [255,255,255]
  #    ...
  #    [255,255,255]
  #  ], <- len = 1920(num of pixels of row)
  #  ...
  #  [
  #    [255,255,255]
  #    [255,255,255]
  #    ...
  #    [255,255,255]
  #  ], <- len = 1920(num of pixels of row)
  # ]

  img1_bgr = image.load(img1src)
  img2_bgr = image.load(img2src)

  
  img1 = image.apply_bgr2hsv(img1_bgr)
  img2 = image.apply_bgr2hsv(img2_bgr)

  # img1 = image.apply__filter_gaussian_blur(image.apply_clahe(image.apply_bgr2gray(img1)))
  # img2 = image.apply__filter_gaussian_blur(image.apply_clahe(image.apply_bgr2gray(img2)))

  # result = image.get_absdiff(img1, img2)
  # image.show(result, "absdiff_example")
  # return

  DIFF_D_MANHATTAN_THRESHOLD = 100.0 # 差分と検出する閾値
  result = generate_allzero_uint8_nparr(1920, 1080)

  for i_col in range(img1.__len__()):
    img1row = img1[i_col]
    img2row = img2[i_col]

    for i_row in range(img1row.__len__()):
      img1px = img1row[i_row]
      img2px = img2row[i_row]

      print(img1px, img2px)
      diff_d_manhattan = np.linalg.norm(img1px - img2px, ord=1)

      if diff_d_manhattan > DIFF_D_MANHATTAN_THRESHOLD:

        # 差分が白だったら消す
        # if img2px[0] < 240 and img2px[1] < 240 and img2px[2] < 240:
          result[i_col, i_row] = img2px


        # result.append([
        #   i_row, # position of x
        #   i_col, # position of y
        #   diff_d_manhattan, # size of diff
        #   img1px, # color mapping of x
        #   img2px
        #   ])

  # pprint(result)
  image.show(result,"result")

# image = CV2ImageUtil()
# j = 0
# for i in range(len(result_img) - 1):
#   # prev_frame_gray = image.apply_grayscale(result_img[i])
#   # current_frame_gray = image.apply_grayscale(result_img[i+1])

#   image.save(result_img[i], os.path.join(data_base_dir, f"frame_"))
  
#   j = j+1
#   if j> 3:
#     return


compare_2_imgs(os.path.join(data_base_dir,"66.jpg"), os.path.join(data_base_dir,"67.jpg"))