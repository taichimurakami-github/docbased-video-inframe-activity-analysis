import cv2
import numpy as np
import pyocr
import pyocr.builders
from collections.abc import Iterable

def generate_allzero_uint8_nparr(width: int, height: int):
    result = []
    for i_col in range(height):
        px_col = []
        for i_row in range(width):
            px_col.append([0, 0, 0])

        result.append(px_col)

    return np.array(result, dtype="uint8")


def detect_imdiff_by_saturation(
    img1_hsv: cv2.Mat,
    img2_hsv: cv2.Mat,
    TH_VALID_SATURATION_DIFF=5,
    img_res=(1920, 1080),
):
    result_diff_img_hsv = generate_allzero_uint8_nparr(img_res[0], img_res[1])
    result_details = []

    for i_col in range(img1_hsv.__len__()):
        img1row = img1_hsv[i_col]
        img2row = img2_hsv[i_col]

        for i_row in range(img1row.__len__()):
            img1px_s = img1row[i_row][1]
            img2px_s = img2row[i_row][1]

            # 変化前後のpxの彩度差が一定以下の場合はノイズとして除去
            saturation_diff_abs = abs(int(img1px_s) - int(img2px_s))
            if saturation_diff_abs < TH_VALID_SATURATION_DIFF:
                continue

            # Update result image
            result_diff_img_hsv[i_col, i_row] = np.array(
                img2row[i_row], dtype="uint8"
            )

            # Update reslut details
            result_details.append(
                [
                    (
                        i_row,
                        i_col,
                        saturation_diff_abs,
                    ),  # position of detected diff pixel
                    img1row[i_row],
                    img2row[i_row],
                ]
            )

    return np.array(result_diff_img_hsv, dtype="uint8"), result_details


def detect_imdiff_by_bgrvalue(
    img1_bgr: cv2.Mat,
    img2_bgr: cv2.Mat,
    TH_VALID_BGRVALUE_MANHATTAN_DIFF=100,
    img_res=(1920, 1080),
):
    result_diff_img_bgr = generate_allzero_uint8_nparr(img_res[0], img_res[1])
    result_details = []

    for i_col in range(img1_bgr.__len__()):
        img1row = img1_bgr[i_col]
        img2row = img2_bgr[i_col]

        for i_row in range(img1row.__len__()):
            img1px = img1row[i_row]
            img2px = img2row[i_row]

            diff_d_manhattan = np.linalg.norm(img1px - img2px, ord=1)

            if diff_d_manhattan > TH_VALID_BGRVALUE_MANHATTAN_DIFF:
                # 差分が白だったら消す
                if img2px[0] < 240 and img2px[1] < 240 and img2px[2] < 240:
                    # Update result image
                    result_diff_img_bgr[i_col, i_row] = np.array(
                        img2px, dtype="uint8"
                    )

                    # Update reslut details
                    result_details.append(
                        [
                            i_row,  # position of x
                            i_col,  # position of y
                            diff_d_manhattan,  # size of diff
                            img1px,  # color mapping of x
                            img2px,
                        ]
                    )

    return np.array(result_diff_img_bgr, dtype="uint8"), result_details


# https://blog.machine-powers.net/2018/08/04/pyocr-and-tips/
class OcrTextExtractor:
    tool = None
    builder = None
    language = ""

    def __init__(
        self,
        tesseractPath: str,
    ):
        pyocr.tesseract.TESSERACT_CMD = tesseractPath
        tools = pyocr.get_available_tools()

        if len(tools) == 0:
            raise Exception("ERROR: No OCR tool found")

        self.tool = tools[0]

    # setting languages
    def use_japanese_lang(self):
        self.language = "jpn"
        return self

    def use_english_lang(self):
        self.language = "eng"
        return self

    # Setting builders
    def use_word_box_builder(self):
        self.builder = pyocr.builders.WordBoxBuilder()
        return self

    def use_linebox_builder(self):
        self.builder = pyocr.builders.LineBoxBuilder()
        return self

    def write_extracted_result_as_hOCR_fmt(self, filepath, result):
        with open(filepath, "w", encoding="utf-8") as f:
            self.builder.write_file(f, result)

    # pyocrではpilowのImageオブジェクトを使用する
    # https://note.com/djangonotes/n/ne993a087f678
    def extract(self, pil_image):
        return self.tool.image_to_string(
            pil_image, lang=self.language, builder=self.builder
        )


# Pyocr official README.md (gitlab) を参照に，builderの生成結果のオブジェクトを解析する
# https://gitlab.gnome.org/World/OpenPaperwork/pyocr
class Img2StrResultParser:
    # Linebox object: list of line objects. For each line object:
    #
    #   line.word_boxes is a list of word boxes (the individual words in the line)
    #   line.content is the whole text of the line
    #   line.position is the position of the whole line on the page (in pixels)
    #
    # print(pyocr.builders.LineBox)
    # >>>
    # [
    #   so 131 190 147 200
    #   that 154 186 183 200
    #   producing 190 186 269 204
    #   50k 276 185 305 200
    #   samples 312 186 375 204
    #   takes 382 186 421 200
    #   approximately 429 185 542 204
    #   5 550 186 557 200
    #   days 565 186 600 204
    # ] 131 185 600 204
    #
    # formats:
    # - wordbased_fmt : list
    # - lineboxbased_fmt : list
    #
    @staticmethod
    def convert_linebox_iterable_into_wordbased_fmt(
        lineboxObjectIterable: Iterable[pyocr.builders.LineBox],
    ):
        jsonSrc = []
        for lineboxObject in lineboxObjectIterable:
            jsonSrc.append(
                {
                    "linePosition": lineboxObject.position,
                    "wordBoxes": list(
                        map(
                            lambda box: [
                                box.content,
                                box.position[0][0],
                                box.position[0][1],
                                box.position[1][0],
                                box.position[1][1],
                            ],
                            lineboxObject.word_boxes,
                        )
                    ),
                }
            )
        return jsonSrc

    @staticmethod
    def convert_linebox_iterable_into_linebased_fmt(
        lineboxObjectIterable: Iterable[pyocr.builders.LineBox],
    ):
        result = []
        for lineboxObject in lineboxObjectIterable:
            result.append(
                {
                    "linePosition": lineboxObject.position,
                    "content": " ".join(
                        map(
                            lambda wordbox: wordbox.content,
                            lineboxObject.word_boxes,
                        )
                    ),
                }
            )
        return result