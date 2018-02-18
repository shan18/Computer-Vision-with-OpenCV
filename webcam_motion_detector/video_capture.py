import cv2
import time


''' Capture the video.
    To use a pre-saved video, provide the video name
    To use a camera, give an integer >= 0. If there are multiple cameras, each camera will have an index.
'''
video = cv2.VideoCapture(0)

frames_generated = 0

# Repeatedly capture all the frames to generate a video
while True:
    frames_generated += 1

    # Read a frame of the video, returns a boolean and a numpy array
    check, frame = video.read()
    print(check)
    print(frame)

    # convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # use sleep if there is no loop and only one frame is displayed
    # time.sleep(3)

    # Display the frame
    cv2.imshow('capture', gray_frame)

    key = cv2.waitKey(1) & 0xFF  # the and operation converts keystroke to ascii value
    if key == ord('q'):  # If key 'q' is pressed, break the loop
        break

print('Frames Generated:', frames_generated)
video.release()  # close the video
cv2.destroyAllWindows()
