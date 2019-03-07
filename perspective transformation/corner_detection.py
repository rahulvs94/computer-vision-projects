import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--image_name", required=True, help="name of the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image_name'])
img = cv2.resize(image, (400, 600))
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 20, 200)

# adjust the kernel size
dst = cv2.cornerHarris(edged, 3, 7, 0.05)

dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
dst = np.uint8(dst)
_, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

for i in range(1, len(corners)):
    cv2.circle(img, tuple(list(map(int, corners[i]))), 3, (0, 0, 255), 3)

oldPoints = np.float32([corners[2], corners[1], corners[3], corners[4]])

newPoint = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

matrix = cv2.getPerspectiveTransform(oldPoints, newPoint)
result = cv2.warpPerspective(orig, matrix, (400, 600))

cv2.imshow('image', img)
cv2.imshow('edged', edged)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
