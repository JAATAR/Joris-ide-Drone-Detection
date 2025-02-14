import cv2
import numpy as np

cap = cv2.VideoCapture
if not cap.isOpened():
    print("Error opening video stream or file")

circle_params = {
    'dp': 1,
    'minDist': 1000,   # Increase for well-separated circles
    'param1': 500,     # Increase for better edge detection
    'param2': 40,      # Increase for better circle identification
    'minRadius': 50,   # Decrease for smaller circles
    'maxRadius': 2000   # Increase for larger circles
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform circle detection on the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blurred = cv2.blur(frame_gray, (3, 3))
    circles = cv2.HoughCircles(frame_blurred, cv2.HOUGH_GRADIENT, **circle_params)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for pt in circles[0, :]:
            a, b, r = int(pt[0]), int(pt[1]), int(pt[2])

            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

    # Display the frame
    cv2.imshow('Circle Detection', frame)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()