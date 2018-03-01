import cv2
import numpy as np
import imutils
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--puzzle', required=True, help='Path to puzzle image.')
args = vars(ap.parse_args())

# load the puzzle and waldo image
puzzle = cv2.imread(args['puzzle'])
waldo = cv2.imread('images/waldo.jpg')
(waldo_height, waldo_width) = waldo.shape[:2]

# find waldo in the puzzle
# we use correlation coefficient matching method
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
(_, _, min_loc, max_loc) = cv2.minMaxLoc(result)

# get roi from puzzle image
top_left = max_loc
bottom_right = (top_left[0] + waldo_width, top_left[1] + waldo_height)
roi = puzzle[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# construct a darkened transparent 'layer'
mask = np.zeros(puzzle.shape, dtype='uint8')
# mask contributes 75% to the darkened image, addWeighted() adds the transparency effect
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

# put the original waldo back in the image to brighten it
puzzle[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

# display
cv2.imshow('Puzzle', imutils.resize(puzzle, height=650))
cv2.imshow('waldo', waldo)
cv2.waitKey(0)
