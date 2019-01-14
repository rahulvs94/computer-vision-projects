import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('lena.tiff')
# image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# image = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

Z = np.float32(image)

K = 1
# number of values for K
values = 6

plt.subplot(1, values+1, 1)
plt.imshow(image, cmap="gray")
plt.xlabel('original image')
plt.xticks([])
plt.yticks([])

for i in range(1, values+1):
    K = K*2
    ret, labels, image_classified = cv2.kmeans(data=Z, K=K, bestLabels=None,
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                               attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    center1 = np.uint8(image_classified)
    res1 = center1[labels.flatten()]
    output = res1.reshape(image.shape)

    plt.subplot(1, values+1, i+1)
    plt.imshow(output, cmap="gray")
    plt.xlabel('K = ' + str(K))
    plt.xticks([])
    plt.yticks([])

'''cv2.imshow("Original image", image)
cv2.imshow("Quantized image", output1)'''


plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
