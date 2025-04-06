import cv2
from ultralytics import YOLO
import numpy as np

window_width = 1280
window_height = 720
cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Video", window_width, window_height)

# identify_colors
def identify_colors(mask,image, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            print(f"{color_name} Contour Area:", cv2.contourArea(contour))

        largest_contour = max(contours, key=cv2.contourArea)  # 找最大輪廓
        largest_area = cv2.contourArea(largest_contour)  # 計算最大輪廓的面積
        x, y, w, h = cv2.boundingRect(largest_contour)  # 獲取邊界框
        aspect_ratio = w / float(h)  # 計算長寬比
        if 0.8 <= aspect_ratio <= 1.3:
            if color_name == "Red" or color_name == "Yellow":
                if largest_area > 500:
                    return color_name, largest_area
            elif color_name == "Green":
                if largest_area > 150:
                    results = model5(image)
                    ArrowSequence = ["<", "^", ">"] #表示三種class的方向
                    for result in results:
                        if hasattr(result, 'probs') and result.probs is not None:
                            confidences = [round(prob.item(), 4) for prob in result.probs.data]
                            predicted_class = result.probs.top1  # 最高概率的类别索引
                            max_confidence = confidences[predicted_class]  # 最高置信度
                            ans = ArrowSequence[predicted_class]  # 对应的类别名称

                            # 打印所有置信度
                            print("All confidences:")
                            for idx, conf in enumerate(confidences):
                                class_name = ArrowSequence[idx]  # 假设 ArrowSequence 是类别名称列表
                                print(f"  {class_name}: {conf}")

                            # print(f"Predicted Class: {ans}, Confidence: {max_confidence}")
                            confidence = round(result.probs.data[predicted_class].item(), 2)  # 直接保留兩位小數
                            ans = ArrowSequence[predicted_class]
                            print(f"Predicted Class: {ans}, Confidence: {confidence}")
                            # cv2.imshow("Green Pre", mask)
                            # cv2.waitKey(0)
                            # cv2.destroyWindow("Green Pre")
                            if confidence > 0.7:
                                return ans, confidence
                        else:
                            print("No classification results found.")
                            return "X", 0.0

                elif largest_area > 400:  # 過濾雜訊
                    return color_name, largest_area
    return None, 0

# split_box
def split_traffic_light(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    split_traffic_light_frames=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 過濾掉小雜訊
            x, y, w, h = cv2.boundingRect(contour)  # 計算外接矩形
            cropped_img = image[y:y+h, x:x+w]  # 裁剪出物件
            split_traffic_light_frames.append(cropped_img)   
    return split_traffic_light_frames

# find_traffic_light
def find_traffic_light(image,c,conf):
    windownumber=f"traffic_light_{c+1}"

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 建立遮罩（包含兩個範圍）
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    split_traffic_light_frames = []
    split_traffic_light_frames+=(split_traffic_light(image, mask_red))
    split_traffic_light_frames+=(split_traffic_light(image, mask_green))
    split_traffic_light_frames+=(split_traffic_light(image, mask_yellow))

    light_colors = []

    # 轉換為 HSV 色彩空間
    for i, frame in enumerate(split_traffic_light_frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 建立遮罩（包含兩個範圍）
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 先嘗試識別不同燈號
        red_result = identify_colors(mask_red,frame, "Red")
        yellow_result = identify_colors(mask_yellow,frame, "Yellow")
        green_result = identify_colors(mask_green,frame, "Green")

        # 判斷目前燈號
        light_BGR = (255, 255, 255)
        largest_area = 0

        if red_result[0]:  # 檢查是否識別到紅燈
            light_colors.append(red_result[0])
            light_BGR = (0, 0, 255)
            largest_area = red_result[1]
        elif yellow_result[0]:  # 檢查是否識別到黃燈
            light_colors.append(yellow_result[0])
            light_BGR = (0, 255, 255)
            largest_area = yellow_result[1]
        elif green_result[0]:  # 檢查是否識別到綠燈
            light_colors.append(green_result[0])
            light_BGR = (0, 255, 0)
            largest_area = green_result[1]

        print(f"目前的紅綠燈狀態：{light_colors}")

        image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(image, f"{conf}, {largest_area}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_BGR, 2)
        # last frame
        if i==len(split_traffic_light_frames)-1:
            for x, light_color in enumerate(light_colors,0):
                # 根據 light_colors 設定文字顏色
                if light_color == "Red":
                    light_BGR = (0, 0, 255)  # 紅色
                elif light_color == "Yellow":
                    light_BGR = (0, 255, 255)  # 黃色
                elif light_color == "Green" or light_color == ">":
                    light_BGR = (0, 255, 0)  # 綠色
                else:
                    light_BGR = (255, 255, 255)  # 綠色
                cv2.putText(image, f"{light_color}", (15*x, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_BGR, 2)

    cv2.imshow(f"{windownumber}", image)
    cv2.moveWindow(f"{windownumber}",c*300+100,100)

    
# yoloVideo
def video(image):
    result = model.predict(image, classes=[0, 1, 2, 3, 5, 7, 9])
    cv2.imshow("Original Video",result[0].plot())


    traffic_light_class = 9  
    traffic_light_count = 0 
    conflist=[-1,-1,-1]
    boxlist=[-1,-1,-1]
    for box in result[0].boxes:
        cls = int(box.cls)  # 取得類別索引
        conf = float(box.conf)  # 取得該物件的信心值
        if cls == traffic_light_class and conf > 0.65:  # 只處理 traffic light
            if conf>conflist[0]: # 找信心值前三大
                conflist[2]=conflist[1]
                conflist[1]=conflist[0]
                conflist[0]=conf
                boxlist[2]=boxlist[1]
                boxlist[1]=boxlist[0]
                boxlist[0]=box
            elif conf>conflist[1]:
                conflist[2]=conflist[1]
                conflist[1]=conf
                boxlist[2]=boxlist[1]
                boxlist[1]=box
            elif conf>conflist[2]:
                conflist[2]=conf
                boxlist[2]=box

    c=0
    for conf in conflist:
        if(conf!=-1):
            box=boxlist[c]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得邊界框座標
            find_traffic_light(image[y1:y2, x1:x2],c,round(conflist[c],2))  # 截圖紅綠燈
            traffic_light_count += 1
        else:
            if cv2.getWindowProperty(f"traffic_light_{c+1}",cv2.WND_PROP_VISIBLE)==1:
                cv2.destroyWindow(f"traffic_light_{c+1}")
        c+=1
           
    print(f"總共擷取 {traffic_light_count} 個紅綠燈。")


# Startup
model = YOLO("YOLOv8l.pt")
# model5 = YOLO("best_ARROW.pt").to("cuda")
# model5 = YOLO("bestSingleArrowE40.pt")
model5 = YOLO("bestSingleArrowTrain12.pt")
target_classes = [0, 1, 2, 3, 5, 7, 9]  
# print(model.names)

# 定義紅、黃、綠燈的 HSV 範圍
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
lower_yellow, upper_yellow = np.array([12, 100, 20]), np.array([35, 255, 255])
lower_green, upper_green = np.array([65, 155, 90]), np.array([90, 255, 255])
# lower_green = np.array([40, 100, 100])
# upper_green = np.array([90, 255, 255])

# 開啟影片
# video_path = "video/2025-01-19_11-47-10-front - Trim.mp4"
# video_path = "video/color_changed.mp4"
video_path = "video/arrow_test.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0  # 計數器
paused = False  # 設定一個變數來追蹤是否暫停
while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 == 0:  # 每 3 幀才進行 YOLO 偵測
            video(frame)

    key = cv2.waitKey(30) & 0xFF  # 只執行一次

    if key == 27:  # 按 'ESC' 退出
        break
    elif key == 32:  # 按 '空白鍵' 暫停/繼續
        paused = not paused
        while paused:  # 停留在這裡，直到使用者按下空白鍵繼續
            key = cv2.waitKey(30) & 0xFF
            if key == 32:
                paused = False
            elif key == 27:
                paused = False
                break

cap.release()
cv2.destroyAllWindows()
