import json
import Levenshtein
import difflib
from enum import Enum, auto


class MATCHING_METHOD(Enum):
    PATTERN_MATCH = auto()
    SEQUENCE_MATCH = auto()
    LEVENSHTEIN_DIST = auto()
    JARO_WINKLER = auto()
    N_GRAM = auto()


class MatchingEngine:
    _base_LBbased_fmt = None
    m_detail_log = []
    _TH_VALID_LEVENSHTEIN = 10
    _TH_VALID_N_GRAM = 0.8

    def __init__(self, lineboxbased_fmt_base_json_path: str):
        with open(lineboxbased_fmt_base_json_path, "r") as f:
            self._base_LBbased_fmt = json.load(f)

    def __perform_pattern_match(self, text1: str, text2: str):
        return text1 == text2

    def __perform_sequence_match(self, text1: str, text2: str):
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def __perform_levenshtein_dist(self, text1: str, text2: str):
        return Levenshtein.distance(text1, text2)

    def __perform_jaro_winkler(self, text1: str, text2: str):
        return Levenshtein.jaro_winkler(text1, text2)

    def __perform_n_gram(self, text1: str, text2: str):
        # splitting text by 2-words (for bi-gram)
        text1_list = []
        for i in range(len(text1) - 1):
            text1_list.append(text1[i : i + 2])

        text2_list = []
        for i in range(len(text2) - 1):
            text2_list.append(text2[i : i + 2])

        total_check_count = 0
        equal_count = 0
        for text1_word in text1_list:
            total_check_count = total_check_count + 1
            equal_flag = 0
            for text2_word in text2_list:
                if text1_word == text2_word:
                    equal_flag = 1
            equal_count = equal_count + equal_flag

        if total_check_count <= 0:
            return 0

        return equal_count / total_check_count

    def __write_matching_logs(self, target_LBbased_fmt, base_i_start):
        (
            base_current_content,
            base_next_content,
            base_after_next_content,
        ) = self.__get_3times_series_linebox_content(
            self._base_LBbased_fmt, base_i_start
        )
        (
            target_current_content,
            target_next_content,
            target_after_next_content,
        ) = self.__get_3times_series_linebox_content(target_LBbased_fmt)

        return self.m_detail_log.append(
            {
                "current": [
                    base_current_content,
                    target_current_content,
                    self.__perform_pattern_match(
                        base_current_content, target_current_content
                    ),
                    self.__perform_sequence_match(
                        base_current_content, target_current_content
                    ),
                    self.__perform_levenshtein_dist(
                        base_current_content, target_current_content
                    ),
                    self.__perform_jaro_winkler(
                        base_current_content, target_current_content
                    ),
                    self.__perform_n_gram(
                        base_current_content, target_current_content
                    ),
                ],
                "next": [
                    base_next_content,
                    target_next_content,
                    self.__perform_pattern_match(
                        base_next_content, target_next_content
                    ),
                    self.__perform_sequence_match(
                        base_next_content, target_next_content
                    ),
                    self.__perform_levenshtein_dist(
                        base_next_content, target_next_content
                    ),
                    self.__perform_jaro_winkler(
                        base_next_content, target_next_content
                    ),
                    self.__perform_n_gram(
                        base_next_content, target_next_content
                    ),
                ],
                "after_next": [
                    base_after_next_content,
                    target_after_next_content,
                    self.__perform_pattern_match(
                        base_after_next_content, target_after_next_content
                    ),
                    self.__perform_sequence_match(
                        base_after_next_content, target_after_next_content
                    ),
                    self.__perform_levenshtein_dist(
                        base_after_next_content, target_after_next_content
                    ),
                    self.__perform_jaro_winkler(
                        base_after_next_content, target_after_next_content
                    ),
                    self.__perform_n_gram(
                        base_after_next_content, target_after_next_content
                    ),
                ],
            }
        )

    def __get_3times_series_linebox_content(self, LBbased_fmt, i_start=0):
        return (
            LBbased_fmt[i_start]["content"],
            LBbased_fmt[i_start + 1]["content"],
            LBbased_fmt[i_start + 2]["content"],
        )

    def compare_lineboxbased_fmt(
        self, LBbased_matching_target, method: MATCHING_METHOD
    ):
        (
            target_current_content,
            target_next_content,
            target_after_next_content,
        ) = self.__get_3times_series_linebox_content(LBbased_matching_target)

        for i in range(len(self._base_LBbased_fmt) - 2):
            (
                base_current_content,
                base_next_content,
                base_after_next_content,
            ) = self.__get_3times_series_linebox_content(
                self._base_LBbased_fmt, i
            )

            # Record matching details
            self.__write_matching_logs(LBbased_matching_target, i)

            match method:
                case MATCHING_METHOD.PATTERN_MATCH:
                    if (
                        self.__perform_pattern_match(
                            base_current_content, target_current_content
                        )
                        and self.__perform_pattern_match(
                            base_next_content, target_next_content
                        )
                        and self.__perform_pattern_match(
                            base_after_next_content, target_after_next_content
                        )
                    ):
                        return self._base_LBbased_fmt[i]

                case MATCHING_METHOD.LEVENSHTEIN_DIST:
                    if (
                        self.__perform_levenshtein_dist(
                            base_current_content, target_current_content
                        )
                        < self._TH_VALID_LEVENSHTEIN
                        and self.__perform_levenshtein_dist(
                            base_next_content, target_next_content
                        )
                        < self._TH_VALID_LEVENSHTEIN
                        and self.__perform_levenshtein_dist(
                            base_after_next_content, target_after_next_content
                        )
                        < self._TH_VALID_LEVENSHTEIN
                    ):
                        return self._base_LBbased_fmt[i]

                case MATCHING_METHOD.N_GRAM:
                    if (
                        self.__perform_n_gram(
                            base_current_content, target_current_content
                        )
                        > self._TH_VALID_N_GRAM
                        and self.__perform_n_gram(
                            base_next_content, target_next_content
                        )
                        > self._TH_VALID_N_GRAM
                        and self.__perform_n_gram(
                            base_after_next_content, target_after_next_content
                        )
                        > self._TH_VALID_N_GRAM
                    ):
                        return (
                            self._base_LBbased_fmt[i],
                            LBbased_matching_target[0],
                        )

                case _:
                    pass

        return ()
