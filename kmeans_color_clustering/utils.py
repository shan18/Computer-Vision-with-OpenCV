import cv2
import numpy as np


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype('float')
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency of each color
    # define a 300×50 pixel rectangle to hold the most dominant colors in the image
    bar = np.zeros((50, 300, 3), dtype='uint8')
    start_x = 0

    # loop over the percentage of each cluster and the color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + percent * 300
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color.astype('uint8').tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar
