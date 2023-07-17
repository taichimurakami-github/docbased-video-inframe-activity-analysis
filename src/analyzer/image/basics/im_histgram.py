import cv2
import numpy as np
from matplotlib import pyplot as plt


def plt_hsv_histgrams(
    cv_img_hsv,
    title="",
    color=("g", "r", "y"),
    plt_label=("Hue", "Saturation", "Value"),
    plt_xlim=[0, 256],
):
    plt.figure(figsize=(10, 6))
    histogram = []  # list[np.ndarray(float32)]

    for i, col in enumerate(color):
        histr = cv2.calcHist([cv_img_hsv], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, label=plt_label[i])
        histogram.append(histr[:, 0])

    plt.xlim(plt_xlim)
    plt.title(label=title)
    plt.legend()
    plt.show()

    print(type(histogram[0][0]))
    print(f"==== Hue ====")
    print(np.sort(histogram[0])[::-1])
    print("==== Saturation ====")
    print(np.sort(histogram[1])[::-1])
    print("==== Value ====")
    print(np.sort(histogram[2])[::-1])

    plt.clf()

    return histogram
