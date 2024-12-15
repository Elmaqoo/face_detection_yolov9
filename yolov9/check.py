import cv2
import numpy as np

def run(source='../assets/video/test_4.mp4'):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Using a more compatible codec and container (MJPG in AVI)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    # Test write a simple frame
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if not out.write(test_frame):
        print("Failed to write test frame.")
    else:
        print("Test frame written successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame.")
            break

        # Optionally resize or process the frame
        # frame = cv2.resize(frame, (width, height))  # Ensure frame size matches expected

        if not out.write(frame):
            print("Error writing frame.")
        else:
            print("Frame written successfully.")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(cv2.__version__)

if __name__ == "__main__":
    run()
