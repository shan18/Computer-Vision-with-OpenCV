import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Read the image in rgb format for display purposes
# Use grayscale version for face detection as it increases the accuracy of openCV
img = cv2.imread('images/news.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# It outputs a numpy ndarray (each 1D array corresponding to a single face) with four values:
#    (1) upper left x-axis pixel value
#    (2) upper left y-axis pixel value
#    (3) box width
#    (4) box height
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print('type:', type(faces))
print('face locations:', faces)

# Loop through all the faces and draw rectangles around them
# rectangle() requires the following parameters":
#    image object,
#    (x,y) coordinates of the upper-left and lower-right corner,
#    border color in (B, G, R) format
#    width of the rectangle
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

plt.imshow(img)
plt.show()
