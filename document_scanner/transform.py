import cv2
import argparse
import numpy as np

from scipy.spatial import distance


def order_points_old(points):
    # Initialize a list of coordinates that will be ordered as following:
    # top-left, top-right, bottom-right and bottom-left
    rect = np.zeros((4, 2), dtype='float32')

    # top-left point will have the smallest sum
    # bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # top-right point will have the smallest difference
    # bottom-left point will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def order_points_euclidean(points):
    # sort the points according to x-coordinate
    x_sorted = points[np.argsort(points[:, 0]), :]

    left_most = x_sorted[:2]
    right_most = x_sorted[2:]

    # sort the left-most points according to y-coordinate
    # to get top-left and bottom-left points
    (tl, bl) = left_most[np.argsort(left_most[:, 1]), :]

    # calculate the euclidean distance of the rightmost points
    # with the top-left point, the one will the max distance will
    # be the bottom right point (pythagoras theorem)
    dist = distance.cdist(tl[np.newaxis], right_most, 'euclidean')[0]
    (tr, br) = right_most[np.argsort(dist), :]

    return np.array([tl, tr, br, bl], dtype='float32')


def four_point_transform(image, points):
    # obtain the points in a ordered fashion
    rect = order_points_euclidean(points)
    (tl, tr, br, bl) = rect

    # calculate width of the new image
    # this will be the maximum distance between top-right and top-left x-coordinates
    # or bottom-right and bottom-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(width_a), int(width_b))

    # calculate height of the new image
    # this will be the maximum distance between top-right and bottom-right y-coordinates
    # or top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(height_a), int(height_b))

    # get the top-down view of the new image with points in the specified order
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # compute and apply the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(image, M, (width, height))

    return warped_img


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image file')
    ap.add_argument('-c', '--coordinates', help='comma separated list of source points')
    args = vars(ap.parse_args())

    img = cv2.imread(args['image'])
    pts = np.array(eval(args['coordinates']), dtype='float32')

    warped = four_point_transform(img, pts)

    cv2.imshow('Original', img)
    cv2.imshow('Warped', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
