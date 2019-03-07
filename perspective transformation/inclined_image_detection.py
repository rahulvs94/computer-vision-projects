import cv2
import imutils
import numpy as np
import argparse

##################### NOTE #####################

# This code would work if it detects approximately 4 points

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--image_name", required=True, help="name of the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image_name'])
image = cv2.resize(image, (400, 600))
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 20, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(approx)

    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
for i in range(len(screenCnt)):
    cv2.circle(image, tuple(screenCnt[i][0]), 3, (0, 0, 255), 2)

screenCnt = screenCnt.reshape(4, 2)
newScreenCnt = np.float32([screenCnt[3], screenCnt[2], screenCnt[0], screenCnt[1]])

newPoint = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

matrix = cv2.getPerspectiveTransform(newScreenCnt, newPoint)
result = cv2.warpPerspective(orig, matrix, (400, 600))

cv2.imshow("Original", image)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
