import cv2
from djitellopy import tello
import time
import numpy as np

# Initialiseren en verbinden met de Tello-drone
me = tello.Tello()
me.connect()
print(f"Batterijpercentage: {me.get_battery()}%")


me.streamon()  # Start de videostream van de drone

# Cirkeldetectie parameters
circle_params = {
    'dp': 1,
    'minDist': 200,  # Pas aan op basis van de grootte van de cirkel en nabijheid van andere cirkels
    'param1': 300,  # Pas aan op basis van de helderheid van de cirkelranden
    'param2': 40,  # Pas aan op basis van de rondheid van het object en het ruisniveau
    'minRadius': 50,  # Pas aan op basis van de verwachte minimale straal van de cirkel
    'maxRadius': 200  # Pas aan op basis van de verwachte maximale straal van de cirkel
}

me.takeoff()
me.move_up(80)

# Pauzeer kort om stabilisatie toe te staan
time.sleep(2)

frame_read = me.get_frame_read()

cv2.imshow("Frame", frame_read.frame)
# Beweeg de drone naar rechts
me.move_right(1000)  # 100 cm naar rechts
for _ in range(3):     me.move_left(100)
# Pauzeer weer kort
time.sleep(2)

while True:
    frame = me.get_frame_read().frame  # Haal het huidige frame van de videostream
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Zet om naar grijstinten
    frame_blurred = cv2.blur(frame_gray, (3, 3))  # Pas blurring toe voor betere detectie

    # Voer cirkeldetectie uit
    circles = cv2.HoughCircles(frame_blurred, cv2.HOUGH_GRADIENT, **circle_params)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for pt in circles[0, :]:
            a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)  # Teken de omtrek van de cirkel
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)  # Teken een klein cirkeltje in het midden

    cv2.imshow("Drone Camera", frame)  # Toon het frame met de gedetecteerde cirkels

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Sluit af als 'q' wordt ingedrukt
        break

me.land()  # Laat de drone landen
me.streamoff()  # Zet de videostream uit
cv2.destroyAllWindows()  # Sluit alle OpenCV-vensters