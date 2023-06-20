import cv2
import os
from enum import Enum ,auto
from utils.CV2VideoUtil import CV2VideoUtil
from utils.CV2ImageUtil import CV2ImageUtil
from analyzer.document import MatchingEngine, MATCHING_METHOD
from analyzer.image import OcrTextExtractor, Img2StrResultParser

PDF_LINEBOXBASED_JSON_SRC = os.path.join(
    os.path.dirname(__file__),
    ".data",
    "pdf",
    "document_index.json",
)
OCR_TOOL_ABS_PATH = "/** Tesseract OCR bin folder path here */"

class ACTIVITY_STATE(Enum):
    ACTIVE = auto()
    DONE = auto()

def analyze_video_activities(video_file_src: str):
    ocr_text_extractor = OcrTextExtractor(OCR_TOOL_ABS_PATH)
    m_engine = MatchingEngine(PDF_LINEBOXBASED_JSON_SRC)
    video = CV2VideoUtil()
    image = CV2ImageUtil()
    cap = video.load(video_file_src)

    _ , video_frames = video.get_video_frame_every_sec(cap)

    def detect_acitvity(
        img1_hsv: cv2.Mat,
        img2_hsv: cv2.Mat,
        TH_VALID_SATURATION_DIFF=5,
        img_res=(1920, 1080),
    ):

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

                return True
        
        return False

    def detect_document_focused(cv2img:cv2.Mat):
        pilimg = image.cvt_to_pil_img(image.apply_bgr2rgb(cv2img))

        extracted_result_lbfmt = Img2StrResultParser.convert_linebox_iterable_into_linebased_fmt(
            ocr_text_extractor
                .use_english_lang()
                .use_linebox_builder()
                .extract(pilimg)
        )

        # マッチング結果が存在する場合は中身入りのlistが返ってくる
        cmp_result = m_engine.compare_lineboxbased_fmt(extracted_result_lbfmt, MATCHING_METHOD.N_GRAM)

        return cmp_result.__len__() > 0

    # activity発生を記録するタイムライン
    tl_activities = [] # [ACTIVITIY_STATE, t_start, t_end][]

    for t_frame in range(video_frames.__len__() - 1):

        prev_img = video_frames[t_frame]
        curr_img = video_frames[t_frame]

        # 比較対象の２フレームはドキュメント内にマッチングする部分があるか判定
        if detect_document_focused(prev_img) == False or detect_document_focused(curr_img) == False:
            continue
        
        # 比較対象の２フレーム間に彩度ベースの解析で差分が認められるかを判定
        if detect_acitvity(prev_img, curr_img) == False:
            continue
        
        # 1. activityの開始を検知
        # 条件：
        # 1-1 直前に終了が記録されている
        # 1-2 tl_activitiyの長さが0

        if tl_activities.__len__() == 0 or tl_activities[-1][0] == ACTIVITY_STATE.DONE:
            # activity開始
            tl_activities.append([ACTIVITY_STATE.ACTIVE, t_frame, -1])
            
        # 2. activityの終了を検知
        elif tl_activities.__len__() > 0 and tl_activities[-1][0] == ACTIVITY_STATE.ACTIVE:
            # activity終了
            tl_activities[-1][0] = ACTIVITY_STATE.DONE
            tl_activities[-1][2] = t_frame

    return tl_activities


if __name__ == "__main__":
    video_file_src = os.path.join(os.path.dirname(__file__), ".data", "EdanMeyerVpt")