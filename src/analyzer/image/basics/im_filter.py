import cv2


def perform_gaussian_filter(imdiff, n_perform=1, kernel=(5, 5)):
    n = n_perform
    imresult = imdiff.copy()

    while n > 0:
        imresult = cv2.GaussianBlur(imresult, kernel, 0)
        n -= 1

    return imresult


def perform_bilateral_filter(imdiff, n_perform=1):
    n = n_perform
    imresult = imdiff.copy()

    while n > 0:
        imresult = cv2.bilateralFilter(imresult, 9, 75, 75)
        n -= 1

    return imresult


def perform_median_filter(imdiff, n_perform=1):
    n = n_perform
    imresult = imdiff.copy()

    while n > 0:
        imresult = cv2.medianBlur(imresult, 3)
        n -= 1

    return imresult
