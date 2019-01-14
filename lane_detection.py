import cv2
import numpy as np

video = cv2.VideoCapture("challenge.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("road_car_view.mp4")
        continue

    frame = cv2.resize(orig_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    low_yellow = np.array([20, 100, 100])
    up_yellow = np.array([30, 255, 255])

    mask_yellow = cv2.inRange(hsv, low_yellow, up_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask = cv2.bitwise_and(gray_image, mask_yw)
    filter_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    #mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(filter_mask, 50, 150)

    lines = cv2.HoughLinesP(edges, 4, np.pi/180, 60, minLineLength=100, maxLineGap=30)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()