import cv2
import numpy as np


# 輪郭に従って切り出すプログラムのテスト
class ImcropAroundContours:
    # 外接矩形の取得
    def __getBoundingRectAngle(self, contour):
        # 配列構造が多重になっているので、１回シンプルな配列にする (point [x y])
        contour = list(map(lambda point: point[0], contour))
        x_list = [point[0] for point in contour]  # width
        y_list = [point[1] for point in contour]  # height
        return [min(x_list), max(x_list), min(y_list), max(y_list)]

    def __getCropData(self, contour, main_axis=0):
        target_axis = 1 if main_axis == 0 else 0

        # axis = 0 = x
        # axis = 1 = y
        contour = list(map(lambda i: i[0], contour))

        axis_point_list = set(
            [point[main_axis] for point in contour]
        )  # unique
        arr = []

        for val in axis_point_list:
            # 輪郭配列から yが特定値のxの配列取得
            target_axis_point_list = list(
                filter(lambda i: i[main_axis] == val, contour)
            )
            tmp_list = [i[target_axis] for i in target_axis_point_list]  #
            arr.append([val, min(tmp_list), max(tmp_list)])
        return arr

    ##y軸に沿ってx座標の範囲を取得していく
    def __doCropY(self, input_im, points, rect):
        height = rect[3] - rect[2]
        width = rect[1] - rect[0]
        left = rect[0]
        top = rect[2]
        output_im = np.zeros((height, width, 3), np.uint8)

        for point in points:
            for x in range(0, width):
                # input画像 座標
                in_y = point[0]
                in_x = x + left
                in_x_min = point[1]
                in_x_max = point[2]

                # output画像座標
                out_y = point[0] - top
                out_x = x
                out_x_min = point[1] - left
                out_x_max = point[2] - left

                # x軸の最大最小の範囲だったら元画像から新画像にコピーする
                if out_x_min < x and x < out_x_max:
                    output_im[
                        out_y : out_y + 1,
                        out_x : out_x + 1,
                    ] = input_im[
                        in_y : in_y + 1,
                        in_x : in_x + 1,
                    ]
        return output_im

    ## 一度抽出済みの画像に対して、
    ## x軸に沿ってy座標の範囲を取得していく
    def __doCropX(self, im, points, rect):
        height = rect[3] - rect[2]
        width = rect[1] - rect[0]
        left = rect[0]
        top = rect[2]

        for point in points:
            for y in range(0, height):
                # input画像 座標
                y = y
                x = point[0] - left
                y_min = point[1] - top
                y_max = point[2] - top
                # y軸の最大最小の範囲だったら元画像から新画像にコピーする
                if y < y_min or y_max < y:
                    im[
                        y : y + 1,
                        x : x + 1,
                    ] = [
                        0,
                        0,
                        0,
                        # 0,
                    ]  # 透過
        return im

    def crop(
        self, contours, im_src: cv2.Mat, th_valid_width=20, th_valid_height=20
    ):
        cropped_img = []
        TH_VALID_WIDTH = th_valid_width
        TH_VALID_HEIGHT = th_valid_height
        TH_VALID_AREA_SIZE = TH_VALID_WIDTH * TH_VALID_HEIGHT

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Check total area of bounding area
            if (
                w > TH_VALID_WIDTH
                and h > TH_VALID_HEIGHT
                and w * h > TH_VALID_AREA_SIZE
            ):
                # 外接矩形の取得
                rect = self.__getBoundingRectAngle(cnt)
                # y座標にxの左端、右端の範囲で切り取る
                crop_data = self.__getCropData(cnt, 1)  # x軸基準
                im_out = self.__doCropY(im_src, crop_data, rect)

                # # #x座標毎にyの上から下の範囲外を透過させる
                crop_data = self.__getCropData(cnt, 0)  # x軸基準
                im_out = self.__doCropX(im_out, crop_data, rect)

                cropped_img.append(crop_data)

        return cropped_img
