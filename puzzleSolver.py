import cv2
import numpy as np


# Helper function to compare to images pixel by pixel using Mean Squared Error
def difference(img1, img2):
    height, weight = img1.shape
    diff = cv2.subtract(img1, img2)
    error = np.sum(diff**2)
    mse = error/float(height*weight)
    return mse


fullImagePath = 'crowd.jpg'
puzzlePiecePath = 'man1.JPG'

cap = cv2.imread(fullImagePath)  # Load image
cap = cv2.resize(cap,(600,600))
piece_cap = cv2.imread(puzzlePiecePath) 

capGray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)    # Apply grayscale
piece_capGray = cv2.cvtColor(piece_cap, cv2.COLOR_BGR2GRAY)

stop_data = cv2.CascadeClassifier('haarcascade_profileface.xml')   # Load the haar-cascade classifier file
foundFaces = stop_data.detectMultiScale(capGray, 1.1, 9)     

print("number if faces recognized: " + str(len(foundFaces)))
for (x, y, w, h) in foundFaces:

    crop_img = capGray[y:y+h, x:x+w]
    piece_capGray = cv2.resize(piece_capGray,(400,400))
    crop_img = cv2.resize(crop_img,(400,400))
    
    mseValue = difference(piece_capGray, crop_img)
    print(mseValue)
    if mseValue < 39:
        cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)


cv2.imshow('Detected faces', cap)   # Display output
cv2.waitKey(0)  

cv2.destroyAllWindows()

