import cv2
import numpy as np
import argparse


# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roi_pts = []
input_mode = False


def select_roi(event, x, y, flags, params):
    # grab the reference to the current frame, list of ROI points
    # and whether or not it is ROI selection mode
    global frame, roi_pts, input_mode

    # if we are in roi selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_pts) < 4:
        roi_pts.append((x, y))
        cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=2)
        cv2.imshow('frame', frame)


def main(params):
    # grab the reference to the current frame, list of ROI points
    # and whether or not it is ROI selection mode
    global frame, roi_pts, input_mode

    # if the video path was not supplied, grab the reference to the camera
    # otherwise, load the video
    if not params.get('video'):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(params['video'])

    # setup the mouse callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', select_roi)

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by atleast one pixel
    # along with the bounding box of roi
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roi_box = None

    # keep looping over the frames
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        print('computing..')

        # check to see if we have reached the end of the video
        if not grabbed:
            break

        # check if the roi has been computed
        if roi_box is not None:
            # convert to HSV color space and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roi_box) = cv2.CamShift(back_proj, roi_box, termination)
            pts = np.int0(cv2.cv.BoxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # show the frame and record if the user presses a key
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'i' key is pressed, go to ROI selection mode
        if key == ord('i') and len(roi_pts) < 4:
            # indicate input mode and clone the frame
            input_mode = True
            orig = frame.copy()

            # keep looping until 4 roi points are selected
            # after selection, press any key to exit
            while len(roi_pts) < 4:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

            # determine the top-left and bottom-right points
            roi_pts = np.array(roi_pts)
            s = roi_pts.sum(axis=1)
            tl = roi_pts[np.argmin(s)]
            br = roi_pts[np.argmax(s)]

            # grab the ROI for the bounding box and convert to hsv color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # compute a hsv histogram and store the bounding box
            roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            roi_box = (tl[0], tl[1], br[0], br[1])
        elif key == ord('q'):
            # if 'q' key is pressed, stop the loop
            break

    # close camera and all other windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', help='path to (optional) video file')
    args = vars(ap.parse_args())

    main(args)
