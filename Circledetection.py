import cv2
import sys
import numpy as np

PREVIEW = 0  # Preview Mode
BLUR = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY = 3  # Canny Edge Detector
CIRCLES = 4

feature_params = dict(maxCorners=500,
                      qualityLevel=0.2,
                      minDistance=15,
                      blockSize=9)

circle_params = {
    'dp': 1,
    'minDist': 500,  # Adjust based on the size of the circle and its proximity to other circles
    'param1': 100,  # Adjust based on the clarity of the circle's edges
    'param2': 30,  # Adjust based on the circularity of the object and noise level
    'minRadius': 10,  # Adjust based on the expected minimum radius of the circle
    'maxRadius': 100  # Adjust based on the expected maximum radius of the circle
}

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):  # Corrected numpy import
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)
    elif image_filter == CIRCLES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blurred = cv2.blur(frame_gray, (3, 3))
        circles = cv2.HoughCircles(frame_blurred, cv2.HOUGH_GRADIENT, **circle_params)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for pt in circles[0, :]:
                a, b, r = int(pt[0]), int(pt[1]), int(pt[2])

                # Draw the circumference of the circle.
                cv2.circle(result, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(result, (a, b), 1, (0, 0, 255), 3)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    elif key == ord('T') or key == ord('t'):
        image_filter = CIRCLES

source.release()
cv2.destroyWindow(win_name)