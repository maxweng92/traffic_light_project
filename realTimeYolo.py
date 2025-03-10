import cv2
from ultralytics import YOLO

target_width = 1280
target_height = 720

# 設定視窗名稱和可調整大小的屬性
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", target_width, target_height)

# 載入模型
model = YOLO("YOLOv8l.pt").to("cuda")
target_classes = [0, 1, 2, 3, 5, 7, 9]  
print(model.names)

# 開啟影片
video_path = "video/2025-01-19_11-47-10-front - Trim.mp4"
video_path = "video/arrow_test.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0  # 計數器

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    if frame_count % 3 == 0:  # 每 3 幀才進行 YOLO 偵測
        results = model(frame, classes=target_classes)
        annotated_frame = results[0].plot()  # 繪製偵測結果
    else:
        annotated_frame = frame  # 直接顯示原始影像

    cv2.imshow("YOLO Detection", annotated_frame)

    # 按 'ESC' 退出
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
