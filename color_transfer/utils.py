import cv2


def rgb_to_lab_transfer(source, target):
    """ Convert the images from the RGB to L*ab* color space,
        being sure to utilizing the floating point data type.
        [Note: OpenCV expects floats to be 32-bit] """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')
    return source, target


def image_stats(image):
    """ Compute the mean and standard deviation of each channel.
        This already assumes that the image is in L*a*b color space. """
    (l, a, b) = cv2.split(image)
    l_mean, l_std = l.mean(), l.std()
    a_mean, a_std = a.mean(), a.std()
    b_mean, b_std = b.mean(), b.std()

    return l_mean, l_std, a_mean, a_std, b_mean, b_std
