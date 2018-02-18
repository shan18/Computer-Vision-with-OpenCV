import sys
import cv2
import imutils
import argparse

from skimage.filters import threshold_local

from transform import four_point_transform


def detect_edges(image):
    """ Edge Detection """
    ratio = image.shape[0] / 500  # compute ratio of old height to new height
    org_image = image.copy()  # save the original copy of the image
    image = imutils.resize(image, height=500)  # resize the image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 100)  # detect edges

    cv2.imshow('Image', image)
    cv2.imshow('Edged', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return org_image, image, edged, ratio


def find_contours(image, edged):
    """ Finding Contours """
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]  # cv2 and cv3 return contours differently
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the approximated contour has four points, then it can be assumed
        # that the contour has been found on our screen
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        print('Upload a proper image.')
        sys.exit(1)

    cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
    cv2.imshow('Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return screen_cnt


def apply_transform_and_threshold(org_image, screen_cnt, ratio):
    """ Obtain bird's eye view of the image
        by applying perspective transform and threshold.
    """
    warped = four_point_transform(org_image, screen_cnt.reshape(4, 2) * ratio)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method='gaussian')
    warped = (warped > T).astype('uint8') * 255  # obtain the black and white (ink on paper) feel

    cv2.imshow('Original', imutils.resize(org_image, height=650))
    cv2.imshow('Scanned', imutils.resize(warped, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image):
    org_image, resized_image, edged, ratio = detect_edges(image)
    screen_cnt = find_contours(resized_image, edged)
    apply_transform_and_threshold(org_image, screen_cnt, ratio)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image file')
    args = vars(ap.parse_args())

    main(cv2.imread(args['image']))
