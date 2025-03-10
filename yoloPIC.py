from ultralytics import YOLO
import cv2

model = YOLO('yolov8l.pt').to('cuda')
# image_path=("pic/2025-01-19_11-47-10-front - frame at 0m2s.jpg") # 0.73 0.73
# image_path=("pic/2025-01-19_11-47-10-front - frame at 0m51s.jpg")
# image_path=("pic/2025-01-19_11-47-10-front - frame at 0m53s.jpg")
# image_path=("pic/2025-01-19_11-47-10-front - frame at 0m56s.jpg")
# image_path=("pic/2025-01-19_11-47-10.jpg")
# image_path=(r"pic\2025-01-19_11-49-10-front - frame at 0m49s.jpg")
image_path=(r"pic\2025-01-19_11-54-11-front - Trim - frame at 0m1s.jpg")
image_path=(r"pic\arrow_test - frame at 0m3s.jpg")

image = cv2.imread(image_path)
result = model(image)
result[0].show()


traffic_light_class = 9  
traffic_light_count = 0 

for box in result[0].boxes:
    cls = int(box.cls)  # 取得類別索引
    conf = float(box.conf)  # 取得該物件的信心值
    if cls == traffic_light_class and conf > 0.7:  # 只處理 traffic light
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得邊界框座標
        cropped_img = image[y1:y2, x1:x2]  # 截圖紅綠燈
        traffic_light_count += 1

        # 儲存圖片
        save_path = f"output/traffic_light_{traffic_light_count}.jpg"
        cv2.imwrite(save_path, cropped_img)
        print((cropped_img))

print(f"總共擷取 {traffic_light_count} 個紅綠燈。")