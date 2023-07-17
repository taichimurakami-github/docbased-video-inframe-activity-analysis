import cv2


def get_area_of_contours(contours):
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        result.append(w * h)

    return result
