import sys
sys.path.insert(0, "../build")
import PyDecoder
import cv2
import time

if __name__ == "__main__":
    video_path = "bigbang.mp4"

    # Test GPU decoder
    dec = PyDecoder.Decoder("GPU:0")
    dec.ingestVideo(video_path)
    dec.decode()
    tick = time.time()
    while True:
        frame = dec.fetchFrame()
        if frame.size == 0:
            break
    tock = time.time()
    print("time taken for gpu video reader is %.3f" % (tock - tick))

    # Test cv2 video capture
    cap = cv2.VideoCapture(video_path)
    tick = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    tock = time.time()
    print("time taken for cv2 video reader is %.3f" % (tock - tick))

