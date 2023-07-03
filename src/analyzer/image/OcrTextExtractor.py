import pyocr
import pyocr.builders


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
