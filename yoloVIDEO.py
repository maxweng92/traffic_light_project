import cv2
from ultralytics import YOLO

# 載入模型
modelVer = "8l"
model = "yolov"+modelVer+".pt"
model = YOLO(model).to("cuda")

# 開啟影片
video_path = "video/2025-01-19_11-47-10-front - Trim2.mp4"
cap = cv2.VideoCapture(video_path)

# 取得影片基本資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 設定輸出影片格式
output_path = "video/11-47-10_output2V"+modelVer+".mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 進行 YOLO 偵測
    results = model(frame)
    annotated_frame = results[0].plot()

    # 寫入輸出影片
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
