import cv2
import numpy as np
from first_order_derivative import main
import scipy.misc
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu, threshold_minimum, threshold_mean
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt

############ Using canny edge detector ############
'''img = cv2.imread('cameraman.tiff')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 75, 150)
noisy_edges = cv2.Canny(gray, 75, 150)
print(type(noisy_edges))

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('gray', gray)
cv2.imshow('edges', edges)
cv2.imshow('without edges', noisy_edges)
cv2.imshow('blur', blur)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

############ Using first order filters (sobel, prewitt, roberts) ############
gray = scipy.misc.imread('wires.jpg', flatten=True)
edges1 = main(gray, name='sobel')
thresh = threshold_otsu(edges1)
binary1 = edges1 > thresh
binary = gaussian_filter(binary1, sigma=0.1, order=1)
print("Otsu's threshold: ", thresh)

lines = probabilistic_hough_line(binary, line_length=5, line_gap=3)

fig, axes = plt.subplots(2, 2)
ax = axes.ravel()

ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Input image')

ax[1].imshow(binary, cmap='gray')
ax[1].set_title('Filtered image')

ax[2].imshow(binary1, cmap='gray')
ax[2].set_title('Sobel edges')

ax[3].imshow(binary * 0)
for line in lines:
    p0, p1 = line
    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[3].set_xlim((0, binary.shape[1]))
ax[3].set_ylim((binary.shape[0], 0))
ax[3].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

