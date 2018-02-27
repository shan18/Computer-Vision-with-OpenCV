import numpy as np
import cv2
import argparse

from utils import rgb_to_lab_transfer, image_stats, show_image


def color_transfer(source, target):
    # compute color statistics for the source and target images
    l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src = image_stats(source)
    l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= l_mean_tar
    a -= l_mean_tar
    b -= b_mean_tar

    # scale by the standard deviations
    l = (l_std_tar / l_std_src) * l
    a = (a_std_tar / a_std_src) * a
    b = (b_std_tar / b_std_src) * b

    # add in the source mean
    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    # clip the pixel intensities to [0, 255]
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to RGB color space
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2RGB)

    return transfer


def main(source, target):
    # load the images
    source = cv2.imread(source)
    target = cv2.imread(target)

    # Convert the images from the RGB to L*ab* color space
    source_lab, target_lab = rgb_to_lab_transfer(source, target)

    # Perform color transfer
    transfer = color_transfer(source_lab, target_lab)

    # show images
    cv2.imshow('source', source)
    cv2.imshow('target', target)
    cv2.imshow('transfer', transfer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', required=True, help='path to source image')
    ap.add_argument('-t', '--target', required=True, help='path to target image')
    args = vars(ap.parse_args())

    main(args['source'], args['target'])
