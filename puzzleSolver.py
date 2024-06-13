import cv2
import numpy as np

fullImagePath = 'Full image.png'
puzzlePiecePath = 'watermellon.JPG'

cap = cv2.imread(fullImagePath, 0)  # Load image
capGray = cv2.cvtColor(cap, cv2.COLOR_BAYER_BG2GRAY)    # Apply grayscale

cv2.imshow("Input Image", capGray)


stop_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   # Load the haar-cascade classifier file
foundFaces = stop_data.detectMultiScale(capGray, 1.1, 9)     

for (x, y, w, h) in foundFaces:
    cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv2.imshow('Detected faces', cap)
cv2.waitKey(0)

cv2.destroyAllWindows()

