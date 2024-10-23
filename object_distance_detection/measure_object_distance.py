from realsense_camera import WebcamCamera
from object_detection import ObjectDetection
import cv2
import numpy as np

camera = WebcamCamera()

# tao phat hien doi tuong dung object_detection
object_detection = ObjectDetection()

# kich thuoc truc x,y
x_range = 100
y_range = 100

# tao toa do mac dinh
ball_coords = (0, 0)  
hoop_coords = (0, 0)

while True:
    
    ret, color_image, _ = camera.get_frame_stream()
    if not ret:
        break  

    height, width = color_image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # Resize cho khung hinh
    resized_image = cv2.resize(color_image, (640, 640))

    
    try:
        bboxes, class_ids, scores = object_detection.detect(resized_image)
    except Exception as e:
        print(f"Lỗi khi phát hiện đối tượng: {e}")
        continue

    # dat co nhan dien ro,bong
    detected_ball = False
    detected_hoop = False

    if bboxes is not None and len(bboxes) > 0 and len(class_ids) > 0 and len(scores) > 0:
        for bbox, class_id in zip(bboxes, class_ids):
            # chuyen toa do tu resize ve khung goc
            x, y, x2, y2 = bbox
            x = int(x * (width / 640))
            y = int(y * (height / 640))
            x2 = int(x2 * (width / 640))
            y2 = int(y2 * (height / 640))

            # ve chu nhat
            cv2.rectangle(color_image, (x, y), (x2, y2), (255, 0, 0), 2)
            # hien thi vat duoc detect
            class_name = object_detection.classes[class_id]
            cv2.putText(color_image, class_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # tinh toa do cua x, y
            obj_center_x = (x + x2) // 2  
            obj_center_y = (y + y2) // 2  

            # chuyen toa do tu pixel sang truc x,y tu -100 - 100
            coord_x = (obj_center_x - center_x) * (x_range / (width // 2))
            coord_y = (center_y - obj_center_y) * (y_range / (height // 2))  
            
            # kiem tra doi tuong
            if class_name == "Banh":  
                ball_coords = (int(coord_x), int(coord_y))
                detected_ball = True  
            elif class_name == "KhungRo":  
                hoop_coords = (int(coord_x), int(coord_y))
                detected_hoop = True  

            # hien thi toa do len khung hinh canh vat the
            text = f"X: {int(coord_x)}, Y: {int(coord_y)}"
            cv2.putText(color_image, text, (x2 + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # neu khong phat hien thi hien toa do mac dinh
    if not detected_ball:
        ball_coords = (0, 0)
    
    if not detected_hoop:
        hoop_coords = (0, 0)

    # Ve truc X
    cv2.line(color_image, (0, center_y), (width, center_y), (0, 255, 0), 2)  # Xanh lá

    # Ve truc y
    cv2.line(color_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # Xanh lá

    # khung hinh hien thi toa do ben trai
    cv2.rectangle(color_image, (10, height - 60), (200, height - 10), (255, 255, 255), -1)
    cv2.putText(color_image, f"Banh - X: {ball_coords[0]}, Y: {ball_coords[1]}",
                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # khung hinh hien thi toa do ben phai
    cv2.rectangle(color_image, (width - 210, height - 60), (width - 10, height - 10), (255, 255, 255), -1)
    cv2.putText(color_image, f"KhungRo - X: {hoop_coords[0]}",
                (width - 200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Color Image", color_image)

    # Exit out
    key = cv2.waitKey(1)
    if key == 27:  
        break

camera.release()
cv2.destroyAllWindows()


