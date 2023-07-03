import cv2
import os
import json
from pprint import pprint
from enum import Enum, auto
from utils.CV2VideoUtil import CV2VideoUtil
from utils.CV2ImageUtil import CV2ImageUtil
from analyzer.document.MatchingEngine import MatchingEngine, MATCHING_METHOD
from analyzer.image.OcrTextExtractor import (
    OcrTextExtractor,
)
from analyzer.image.Img2StrResultParser import Img2StrResultParser
from analyzer.image.detect_imdiff_by_saturation import (
    detect_imdiff_by_saturation,
)
from utils.Benchmark import Benchmark

PDF_LINEBOXBASED_JSON_SRC = os.path.join(
    os.path.dirname(__file__),
    ".data",
    "document_vpt_index.json",
)
OCR_TOOL_ABS_PATH = os.path.join(
    os.path.join(
        os.path.expanduser("~"),
        "AppData",
        "Local",
        "Tesseract-OCR",
        "tesseract.exe",
    )
)


class ACTIVITY_STATE(Enum):
    ACTIVE = auto()
    DONE = auto()


def analyze_video_activities(video_file_src: str):
    benchmark = Benchmark()
    ocr_text_extractor = OcrTextExtractor(OCR_TOOL_ABS_PATH)
    m_engine = MatchingEngine(PDF_LINEBOXBASED_JSON_SRC)
    video = CV2VideoUtil()
    image = CV2ImageUtil()

    i_videoframe_start = 66

    print("\nPreparing for analysis: Loading video")
    cap = video.load(video_file_src)
    _, video_frames = video.get_video_frame_every_sec(cap, i_videoframe_start)
    print("done.")

    def detect_acitvity(
        img1_hsv: cv2.Mat,
        img2_hsv: cv2.Mat,
        TH_VALID_SATURATION_DIFF=5,
        img_res=(1920, 1080),
    ):
        _result_diff_img, result_details = detect_imdiff_by_saturation(
            img1_hsv, img2_hsv
        )
        return result_details.__len__() > 0

        # for i_col in range(img1_hsv.__len__()):
        #     img1row = img1_hsv[i_col]
        #     img2row = img2_hsv[i_col]

        #     for i_row in range(img1row.__len__()):
        #         img1px_s = img1row[i_row][1]
        #         img2px_s = img2row[i_row][1]

        #         # 変化前後のpxの彩度差が一定以下の場合はノイズとして除去
        #         saturation_diff_abs = abs(int(img1px_s) - int(img2px_s))
        #         if saturation_diff_abs < TH_VALID_SATURATION_DIFF:
        #             continue

        #         return True

        # return False

    def detect_document_focused(cv2img: cv2.Mat):
        pilimg = image.cvt_cv_to_pil(cv2img)

        """
        # How to convert PIL image to bin
        image_file = Image.open("convert_iamge.png")

        # 1. convert to grayscale
        image_file = image_file.convert('L') # Grayscale

        # 2. convert pixel value by threshold
        image_file = image_file.point( lambda p: 255 if p > threshold else 0 ) # Threshold

        # 3. convert image state to monochrome
        image_file = image_file.convert('1') # To mono
        """
        pilimg = pilimg.convert("L")
        PIL_THRESH = 200
        pilimg = pilimg.point(lambda p: 255 if p > PIL_THRESH else 0)
        # pilimg.show("test")

        extracted_result_lbfmt = (
            Img2StrResultParser.convert_linebox_iterable_into_linebased_fmt(
                ocr_text_extractor.use_english_lang()
                .use_linebox_builder()
                .extract(pilimg)
            )
        )
        # pprint(extracted_result_lbfmt)

        if extracted_result_lbfmt.__len__() < 3:
            return False

        # マッチング結果が存在する場合は中身入りのlistが返ってくる
        # print("\nExtracted result from current video frame:")
        # pprint(extracted_result_lbfmt)
        cmp_result = m_engine.compare_lineboxbased_fmt(
            extracted_result_lbfmt, MATCHING_METHOD.N_GRAM
        )
        return (
            cmp_result.__len__() > 0,
            cmp_result[0],  # matching result from document_XXX_index.json
            cmp_result[1],  # matching result from current video frame
        )

    def start_activity(tl_activities, t_frame):
        tl_activities.append([ACTIVITY_STATE.ACTIVE, t_frame, -1])

    def end_activity(tl_activities, t_frame):
        tl_activities[-1][0] = ACTIVITY_STATE.DONE
        tl_activities[-1][2] = t_frame

    # activity発生を記録するタイムライン
    tl_activities = []  # [ACTIVITIY_STATE, t_start, t_end][]
    video_length = video.get_media_time_sec(cap)

    for t_frame in range(i_videoframe_start - 1, video_frames.__len__() - 1):
        print(f"\nAnalysis: Frame at t={t_frame} / {video_length}")
        pprint(tl_activities)
        t_currentframe = t_frame + 1
        prev_img = video_frames[t_frame]
        curr_img = video_frames[t_currentframe]

        curr_activity_active = (
            tl_activities.__len__() > 0
            and tl_activities[-1][0] == ACTIVITY_STATE.ACTIVE
        )

        # image.show(curr_img, f"current_frame (t={t_frame})")
        # 現在のフレームを計算してstateを計算
        # 2~3 sec (ave: 2.1~2.5?)
        (
            prevframe_document_matched,
            prevframe_matched_basecontent,
            _prevframe_matched_targetcontent,
        ) = detect_document_focused(prev_img)
        (
            currframe_document_matched,
            currframe_matched_basecontent,
            _currframe_matched_targetcontent,
        ) = detect_document_focused(curr_img)

        # 3~4 sec
        activity_detected = detect_acitvity(prev_img, curr_img)

        print(
            "frame_analysis_result:",
            prevframe_document_matched,
            currframe_document_matched,
            activity_detected,
        )

        if activity_detected == True:
            if (
                curr_activity_active == False
                and prevframe_document_matched == True
                and currframe_document_matched == True
            ):
                start_activity(tl_activities, t_frame)

        else:  # activity_detected == False
            """
            <<activityの終了記録条件>>
            1. 直前のフレームはドキュメント内にマッチするが，現在のフレームはマッチしない場合
            2. 現在のフレームと直前のフレームの間で彩度差分が検知されない = アクティビティが静止した場合
            """
            if (
                (
                    prevframe_document_matched == False
                    and curr_activity_active == True
                )
                or currframe_document_matched == False
                or activity_detected == False
            ):
                end_activity(tl_activities, t_currentframe)

        # 最後の要素がDONE状態で記録されているとき，もしくはアクティビティが存在しない場合，アクティビティの開始を分析

        # 比較対象の２フレーム間に彩度ベースの解析で差分が認められるかを判定
        # if (
        #     t_frame < 53
        #     or (122 <= t_frame <= 131)
        #     or (195 <= t_frame <= 221)
        #     or (278 <= t_frame <= 287)
        #     or detect_acitvity(prev_img, curr_img) == False
        # ):
        # print("Activity: False")

        # 2. activityの終了を検知
        # if (
        #     tl_activities.__len__() > 0
        #     and tl_activities[-1][0] == ACTIVITY_STATE.ACTIVE
        # ):
        #     # activity終了
        #     tl_activities[-1][0] = ACTIVITY_STATE.DONE
        #     tl_activities[-1][2] = t_frame

        # continue

        # 1. activityの開始を検知
        # 条件：
        # 1-1 直前に終了が記録されている
        # 1-2 tl_activitiyの長さが0

        # if (
        #     tl_activities.__len__() == 0
        #     or tl_activities[-1][0] == ACTIVITY_STATE.DONE
        # ):
        #     # activity開始
        #     tl_activities.append([ACTIVITY_STATE.ACTIVE, t_frame, -1])

    return tl_activities


if __name__ == "__main__":
    video_file_src = os.path.join(
        os.path.dirname(__file__), ".data", "EdanMeyerVpt.mp4"
    )
    output_base_dirpath = os.path.join(os.path.dirname(__file__), ".data")

    result = analyze_video_activities(video_file_src)

    with open(
        os.path.join(output_base_dirpath, "activities_timeline.json"), "w"
    ) as fp:
        json.dump(result, fp)
