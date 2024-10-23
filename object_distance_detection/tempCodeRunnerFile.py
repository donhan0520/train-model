from realsense_camera import WebcamCamera
from object_detection import ObjectDetection
import cv2
import numpy as np

# Tạo đối tượng Camera (sử dụng webcam)
camera = WebcamCamera()

# Tạo đối tượng phát hiện đối tượng
object_detection = ObjectDetection()

# Kích thước hệ trục X và Y
x_range = 100
y_range = 100

while True:
    
    ret, color_image, _ = camera.get_frame_stream()
    if not ret:
        break  # Không bắt được khung hình

    height, width = color_image.shape[:2]

    # Vẽ trục X (hoành) và Y (tung) giữa khung hình
    center_x = width // 2
    center_y = height // 2

    # Vẽ trục X từ (-100, 0) đến (100, 0)
    cv2.line(color_image, (0, center_y), (width, center_y), (0, 255, 0), 2)  # Xanh lá

    # Vẽ trục Y từ (0, -100) đến (0, 100)
    cv2.line(color_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # Xanh lá

    # Resize img(640) cho phát hiện đối tượng
    resized_image = cv2.resize(color_image, (640, 640))

    # Phát hiện đối tượng và vẽ thông tin lên hình ảnh
    try:
        bboxes, class_ids, scores = object_detection.detect(resized_image)
    except Exception as e:
        print(f"Lỗi khi phát hiện đối tượng: {e}")
        continue

    ball_coords = (None, None)  # Tọa độ của quả bóng
    hoop_coords = (None, None)  # Tọa độ của rổ

    if bboxes is not None and len(bboxes) > 0 and len(class_ids) > 0 and len(scores) > 0:
        for bbox, class_id in zip(bboxes, class_ids):
            # Chuyển đổi tọa độ từ khung hình resized về khung hình gốc
            x, y, x2, y2 = bbox
            x = int(x * (width / 640))
            y = int(y * (height / 640))
            x2 = int(x2 * (width / 640))
            y2 = int(y2 * (height / 640))

            # Vẽ hình chữ nhật quanh đối tượng
            cv2.rectangle(color_image, (x, y), (x2, y2), (255, 0, 0), 2)
            # Hiển thị tên lớp nếu cần
            class_name = object_detection.classes[class_id]
            cv2.putText(color_image, class_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


            # Tính toán tọa độ dựa trên hệ trục X-Y (gốc giữa khung hình)
            obj_center_x = (x + x2) // 2  # Tọa độ X của tâm vật thể
            obj_center_y = (y + y2) // 2  # Tọa độ Y của tâm vật thể

            # Chuyển đổi tọa độ từ pixel sang giá trị hệ trục (-100 đến 100)
            coord_x = (obj_center_x - center_x) * (x_range / (width // 2))
            coord_y = (center_y - obj_center_y) * (y_range / (height // 2))  # Lưu ý trục Y ngược
            
            # Kiểm tra xem đối tượng là bóng hay rổ
            class_name = object_detection.classes[class_id]
            if class_name == "Banh":  # Giả sử tên lớp của bóng là "ball"
                ball_coords = (int(coord_x), int(coord_y))
            elif class_name == "Ro":  # Giả sử tên lớp của rổ là "hoop"
                hoop_coords = (int(coord_x), int(coord_y))

            # Hiển thị tọa độ bên cạnh khung hình
            text = f"X: {int(coord_x)}, Y: {int(coord_y)}"
            cv2.putText(color_image, text, (x2 + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Vẽ khung chứa tọa độ bóng ở góc dưới bên trái
    if ball_coords != (None, None):
        cv2.rectangle(color_image, (10, height - 60), (200, height - 10), (255, 255, 255), -1)
        cv2.putText(color_image, f"Banh - X: {ball_coords[0]}, Y: {ball_coords[1]}",
                    (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Vẽ khung chứa tọa độ rổ ở góc dưới bên phải
    if hoop_coords != (None, None):
        cv2.rectangle(color_image, (width - 210, height - 60), (width - 10, height - 10), (255, 255, 255), -1)
        cv2.putText(color_image, f"Ro - X: {hoop_coords[0]}, Y: {hoop_coords[1]}",
                    (width - 200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    cv2.imshow("Color Image", color_image)

    # Exit out
    key = cv2.waitKey(1)
    if key == 27:  
        break

# Giải phóng camera
camera.release()
cv2.destroyAllWindows()
