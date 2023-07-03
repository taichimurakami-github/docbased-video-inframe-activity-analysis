from matplotlib import pyplot as plt


def plot_cv_img_bin_with_edges(im_original, im_edges):
    # plot original image
    plt.subplot(121), plt.imshow(im_original, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])

    # plot edged image
    plt.subplot(122), plt.imshow(im_edges, cmap="gray")
    plt.title("Edge Image"), plt.xticks([]), plt.yticks([])

    plt.show()
