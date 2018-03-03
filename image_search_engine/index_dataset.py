import argparse
import cv2
import _pickle
import glob

from utils.rgbhistogram import RGBHistogram


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to the directory that contains the images to be indexed')
ap.add_argument('-i', '--index', required=True, help='Path to where the computed index will be stored')
args = vars(ap.parse_args())


# initialize the index dictionary to store our our quantified
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize the image descriptor -- a 3D RGB histogram with 8 bins per channel
desc = RGBHistogram([8, 8, 8])

# use glob to grab the image paths and loop over them
for image_path in glob.glob(args['dataset'] + '/*.png'):
    # extract the unique image ID (i.e. the filename)
    k = image_path[image_path.rfind('/') + 1:]

    # load the image, describe it using our RGB histogram descriptor, and update the index
    image = cv2.imread(image_path)
    features = desc.describe(image)
    index[k] = features

# after indexing, now write the index to the disk
with open(args['index'], 'wb') as f:
    f.write(_pickle.dumps(index))
