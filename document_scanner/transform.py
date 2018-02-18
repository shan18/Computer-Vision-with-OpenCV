import cv2
import argparse
import numpy as np


def order_points(pts):
    # Initialize a list of coordinates that will be ordered as following:
    # top-left, top-right, bottom-right and bottom-left
    rect = np.zeros((4, 2), dtype='float32')

    # top-left point will have the smallest sum
    # bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point will have the smallest difference
    # bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, points):
    # obtain the points in a ordered fashion
    rect = order_points(points)
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
