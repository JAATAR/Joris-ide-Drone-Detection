from djitellopy import Tello
import cv2
from dbr import *
import numpy as np

circle_params = {
    'dp': 1,
    'minDist': 1000,   # Increase for well-separated circles
    'param1': 500,     # Increase for better edge detection
    'param2': 40,      # Increase for better circle identification
    'minRadius': 50,   # Decrease for smaller circles
    'maxRadius': 2000   # Increase for larger circles
}
def barcode_decoder_setup():
    try:
        # 1.Initialize license.
        # You can also request a 30-day trial license in the customer portal: https://www.dynamsoft.com/customer/license/trialLicense?architecture=dcv&product=dbr&utm_source=samples&package=python
        error = BarcodeReader.init_license("t0068lQAAAGdfv5avdZWA5KMhKvz62eU6+w07YpPydUnp6rwoUamyaVcbQbK5qAVT7qdDlzTITYueeuJ4hgHWacRzer8y1iU=")
        if error[0] != EnumErrorCode.DBR_OK:
            print("License error: " + error[1])

        # 2.Create an instance of Barcode Reader.
        reader = BarcodeReader.get_instance()
        if reader == None:
            raise BarcodeReaderError("Get instance failed")
        # There are two ways to configure runtime parameters. One is through PublicRuntimeSettings, the other is through parameters template.
        # 3. General settings (including barcode format, barcode count and scan region) through PublicRuntimeSettings
        # 3.1 Obtain current runtime settings of instance.
        settings = reader.get_runtime_settings()

        # 3.2 Set the expected barcode format you want to read.
        # The barcode format our library will search for is composed of BarcodeFormat group 1 and BarcodeFormat group 2.
        # So you need to specify the barcode format in group 1 and group 2 individually.
        settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
        settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_POSTALCODE | EnumBarcodeFormat_2.BF2_DOTCODE

        # 3.3 Set the expected barcode count you want to read.
        settings.expected_barcodes_count = 10

        # 3.4 Set the ROI(region of interest) to speed up the barcode reading process.
        # Note: DBR supports setting coordinates by pixels or percentages. The origin of the coordinate system is the upper left corner point.
        settings.region_measured_by_percentage = 1
        settings.region_left = 0
        settings.region_right = 100
        settings.region_top = 0
        settings.region_bottom = 100

        # 3.5 Apply the new settings to the instance
        reader.update_runtime_settings(settings)
        return reader
    except BarcodeReaderError as bre:
        print(bre)

def run_bottom_video(drone, reader):
    while True:
        frame = drone.get_frame_read().frame
        # Read barcodes from the frame
        reader.append_video_frame(frame)

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
        cv2.imshow('Hackathon', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Exiting...")
    drone.end()
    print("Remove object")

def main():
    reader = barcode_decoder_setup()

    tello = Tello()

    tello.connect()

    tello.set_video_direction(tello.CAMERA_DOWNWARD)
    tello.takeoff()

    tello.streamon()
    # Shows the video stream of the drone
    run_bottom_video(tello, reader)

    # Finish the video stream
    tello.land()

    # Release resource
    reader.recycle_instance()


if __name__ == '__main__':
    main()