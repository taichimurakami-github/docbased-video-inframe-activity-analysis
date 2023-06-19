import math
import os
import cv2
from pprint import pprint
from pathlib import Path

class CV2VideoUtil:
  def load(self, video_file_path:str):
    return cv2.VideoCapture(video_file_path)

  def get_media_time_sec(self, cap):
    frame_cnt = self.get_frame_count(cap)
    fps = self.get_fps(cap)
    media_time_sec = math.floor(frame_cnt / fps)

    print(f"frame={frame_cnt}, fps={fps}, total_time={media_time_sec}")

    return media_time_sec

  def get_frame_count(self, cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)

  def get_fps(self, cap):
    return cap.get(cv2.CAP_PROP_FPS)

  def get_video_frame_every_sec(self, cap, start_time_sec = 0):
    media_time_sec = self.get_media_time_sec(cap)
    result_res = []
    result_img = []

    for i in range(media_time_sec + 1): # range : use "<", not "<="

      if i < start_time_sec:
        continue

      cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
      res, img = cap.read()
      result_res.append(res)
      result_img.append(img)

    return result_res, result_img


if __name__ == '__main__':
  video_file_path = os.path.join(Path(__file__).parent.parent.absolute(), ".data", "EdanMeyerVpt.mp4")
  print(video_file_path)

  video = CV2VideoUtil()
  cap = video.load_file(video_file_path)
  _, result_img = video.get_video_frame_every_sec(cap)