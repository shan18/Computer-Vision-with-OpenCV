import cv2


# Read the image, add a integer in params for additional functionality
# 0: grayscale
# 1: RGB format
# -1: enables the transparency functionality
img = cv2.imread('input/galaxy.jpg', 0)

# Get type, dimensions and shape
print('type:', type(img))
print('dimensions:', img.ndim)
print('shape:', img.shape)

# Image display and processing by OpenCV
# waitkey(0) specifies that the image will stay until the user presses any key.
# waitkey(<number grater than 0>) specifies the time in 'ms' during which the image will be displayed.
# destroyAllWindows() tells opencv to destroy all windows when image is closed.
resized_image = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow('test', resized_image)
cv2.imwrite('output/galaxy_resized.jpg', resized_image)  # store the new image with a different name
cv2.waitKey(0)
cv2.destroyAllWindows()
