import cv2
import numpy as np

background1 = cv2.imread("wallpaper.jpg")
background = cv2.resize(background1, (500, 500), interpolation=cv2.INTER_AREA)
foreground1 = cv2.imread("car.jpg")
foreground = cv2.resize(foreground1, (500, 500), interpolation=cv2.INTER_AREA)
foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

### kernel size and filter size for medianBlur will vary by image ###
_, mask = cv2.threshold(foreground_gray, 250, 255, cv2.THRESH_BINARY)
blurred_mask_inv = cv2.medianBlur(mask, 13)

kernel = np.ones((21, 21), np.uint8)
opening = cv2.morphologyEx(blurred_mask_inv, cv2.MORPH_OPEN, kernel)

mask_inv = cv2.bitwise_not(opening)

back = cv2.bitwise_and(background, background, mask=opening)
objects = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
result = cv2.add(back, objects)

#cv2.imshow("background", background)
#cv2.imshow("foreground", foreground)
#cv2.imshow("foreground_gray", foreground_gray)
cv2.imshow("back", back)
cv2.imshow("objects", objects)
cv2.imshow("mask", mask)
#cv2.imshow("mask inverse", mask_inv)
cv2.imshow("blurred mask inv", blurred_mask_inv)
cv2.imshow("opening", opening)
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
