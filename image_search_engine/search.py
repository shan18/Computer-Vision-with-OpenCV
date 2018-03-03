import argparse
import numpy as np
import cv2
import _pickle

from utils.searcher import Searcher


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the indexed images")
ap.add_argument("-i", "--index", required=True, help="Path to the stored index")
args = vars(ap.parse_args())

# load the index and initialize the searcher
index = _pickle.loads(open(args['index'], 'rb').read())
searcher = Searcher(index)

# loop over images in the index - each image will be used as a query image
for (query, query_features) in index.items():
    # perform the search using the current query
    results = searcher.search(query_features)

    # load the query image and display it
    path = args['dataset'] + '/{name}'.format(name=query)
    query_image = cv2.imread(path)
    cv2.imshow('Query', query_image)
    print('query:', query)

    # initialize the two montages to display our results --
    # we have a total of 25 images in the index, but let's only
    # display the top 10 results; 5 images per montage, with
    # images that are 400x166 pixels
    montageA = np.zeros((166 * 5, 400, 3), dtype="uint8")
    montageB = np.zeros((166 * 5, 400, 3), dtype="uint8")

    # loop over the top ten results
    for j in range(10):
        # grab the result (we are using row-major order) and load the result image
        (score, image_name) = results[j]
        path = args['dataset'] + '/{name}'.format(name=image_name)
        result = cv2.imread(path)
        print("\t%d. %s : %.3f" % (j + 1, image_name, score))

        # check to see if the first montage should be used
        if j < 5:
            montageA[j * 166:(j + 1) * 166, :] = result
        # otherwise, the second montage should be used
        else:
            montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

        # show the results
        cv2.imshow("Results 1-5", montageA)
        cv2.imshow("Results 6-10", montageB)
        cv2.waitKey(0)
