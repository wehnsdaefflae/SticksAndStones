import cv2

# Initialize the video capture object.
# The argument can be either the device index or the name of a video file.
# Device index is just the number to specify which camera.
# Normally, one camera will be connected (so we pass 0).
cap = cv2.VideoCapture(0)

# Check if the video capturing object initialized successfully.
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()

# Loop to continuously get frames.
try:
    while True:
        # Read a frame.
        ret, frame = cap.read()

except KeyboardInterrupt:
    cap.release()

finally:
    # Release the capture object and close all OpenCV windows.
    cap.release()
