import pyocr
import pyocr.builders
from collections.abc import Iterable


# https://gitlab.gnome.org/World/OpenPaperwork/pyocr
class Img2StrResultParser:
    # Linebox object: list of line objects. For each line object:
    #
    #   line.word_boxes is a list of word boxes (the individual words in the line)
    #       wordbox.content = "__STRING_HERE__"
    #       wordbox.position = ((421, 877), (452, 896))
    #
    #       ex) > print( wordbox )
    #           >> "__STRING_HERE__" 50 1096 88 1553
    #
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
        default_offset_top=0,
        default_offset_left=0,
        default_offset_bottom=0,
        default_offset_right=0,
    ):
        result = []
        for lineboxObject in lineboxObjectIterable:
            # lineboxObject.position = [ [top,left] , [bottom,right] ]
            position_top = lineboxObject.position[0][0] + default_offset_top
            position_left = lineboxObject.position[0][1] + default_offset_left
            position_bottom = (
                lineboxObject.position[1][1] + default_offset_bottom
            )
            position_right = (
                lineboxObject.position[1][0] + default_offset_right
            )

            result.append(
                {
                    "linePosition": [
                        position_top,
                        position_left,
                        position_bottom,
                        position_right,
                    ],  # bounding-box-like object : [ top, left, bottom, right ]
                    "content": " ".join(
                        map(
                            lambda wordbox: wordbox.content.encode(
                                "cp932", "ignore"
                            ).decode("cp932"),
                            lineboxObject.word_boxes,
                        )
                    ),
                }
            )
        return result
