import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")


def nothing(x):
    pass


cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 182, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 49, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask1, kernel, iterations=1)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 3)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        elif len(approx) == 4:
            cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        elif len(approx) == 5:
            cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
