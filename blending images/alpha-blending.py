import cv2
import numpy as np

# Read the images
foreground1 = cv2.imread("car.jpg")
foreground = cv2.resize(foreground1, (400, 400), interpolation=cv2.INTER_AREA)
background1 = cv2.imread("wallpaper.jpg")
background = cv2.resize(background1, (400, 400), interpolation=cv2.INTER_AREA)

print(foreground.shape)
print(background.shape)

foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

### kernel size and filter size for medianBlur will vary by image ###
_, mask = cv2.threshold(foreground_gray, 253, 255, cv2.THRESH_BINARY)
blurred_mask_inv = cv2.medianBlur(mask, 13)

kernel = np.ones((21, 21), np.uint8)
opening = cv2.morphologyEx(blurred_mask_inv, cv2.MORPH_OPEN, kernel)

mask_inv = cv2.bitwise_not(opening)
alpha = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
print(alpha.shape)

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float)/255
 
# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)
 
# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)
 
# Add the masked foreground and background.
outImage = cv2.add(foreground, background)
 
# Display image
cv2.imshow("foreground", foreground)
cv2.imshow("foreground_gray", foreground_gray)
cv2.imshow("background", background)
cv2.imshow("blurred_mask_inv", blurred_mask_inv)
cv2.imshow("opening", opening)
cv2.imshow("alpha", alpha)
cv2.imshow("outImg", outImage/255)
cv2.imwrite('output image.png', outImage/255)

cv2.waitKey(0)
cv2.destroyAllWindows()
