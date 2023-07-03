import numpy as np
from basics.calculate_projection_profile import (
    calculate_projection_profile_of_bgrimg,
)


def detect_framediff_by_projection_profile(
    cvimg_bgr_prevframe, cvimg_bgr_currframe, th_corr_diff=0.7
):
    (
        h_profile_prevframe,
        v_profile_prevframe,
    ) = calculate_projection_profile_of_bgrimg(cvimg_bgr_prevframe)

    (
        h_profile_currframe,
        v_profile_currframe,
    ) = calculate_projection_profile_of_bgrimg(cvimg_bgr_currframe)

    v_corr = np.corrcoef(h_profile_prevframe, h_profile_currframe)[0, 1]
    h_corr = np.corrcoef(v_profile_prevframe, v_profile_currframe)[0, 1]

    diff_detected = h_corr < th_corr_diff or v_corr < th_corr_diff

    return diff_detected, h_corr, v_corr
