import cv2
import numpy as np

class WebcamCamera:
    def __init__(self):
        print("Loading video")

        # self.cap = cv2.VideoCapture("/home/banana/Documents/RBC_img_analyse/train-model (Copy)/video.mp4")
        self.cap = cv2.VideoCapture(0)

        # Dat do phan giai cho webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit(1)

    def get_frame_stream(self):
        # doc khung hinh tu webcam
        ret, frame = self.cap.read()
        
        
        if not ret:
            print("Error: Could not read frame from webcam.")
            return False, None, None
        
        # Không chuyển đổi sang RGB, giữ nguyên BGR
        color_image = frame  

        return True, color_image, None

    def release(self):
        self.cap.release()