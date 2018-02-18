import os
import cv2

input_path = 'input/batch-images/'
output_path = 'output/'
img_list = os.listdir(input_path)
for img in img_list:
    img_cv2 = cv2.imread(os.path.join(input_path, img), 0)
    img_resize = cv2.resize(img_cv2, (100, 100))
    cv2.imwrite(os.path.join(output_path, img), img_resize)
