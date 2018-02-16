"""
The first frame should be of the static background i.e. without any objects.
"""


import cv2
import pandas
from datetime import datetime


first_frame = None
status_list = [None, None]
timestamp = []
df = pandas.DataFrame(columns=['Start', 'End'])
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0  # no object in frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # See docs: https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    # This is done to reduce noise and increase accuracy
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_frame
        continue

    # Difference between the pixel values of two frames
    delta_frame = cv2.absdiff(first_frame, gray_frame)
    print(delta_frame)

    # pixels with difference greater than 30 will be colored white(255)
    # it returns a tuple with the frame object as the second element
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # this is done smooth out the threshold
    # removes the black holes from the threshold frame
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # draw contours
    (_, contours, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10000:  # to detect bigger objects, give big pixel number
            continue
        status = 1  # object detected
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('object detection', gray_frame)
    cv2.imshow('delta frame', delta_frame)
    cv2.imshow('thresh frame', thresh_frame)
    cv2.imshow('color frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if status == 1:
            timestamp.append(datetime.now())
        break

    status_list.append(status)

    if status_list[-2] == 0 and status_list[-1] == 1:
        timestamp.append(datetime.now())
    elif status_list[-2] == 1 and status_list[-1] == 0:
        timestamp.append(datetime.now())

print('first:')
print(first_frame)

for i in range(0, len(timestamp), 2):
    df = df.append({'Start': timestamp[i], 'End': timestamp[i+1]}, ignore_index=True)
df.to_csv('timestamp.csv')

video.release()
cv2.destroyAllWindows()
