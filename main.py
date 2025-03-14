import cv2
from ultralytics import YOLO
import numpy as np

window_width = 1280
window_height = 720


# 設定視窗名稱和可調整大小的屬性
#cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("YOLO Detection", target_width, target_height)
cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Video", window_width, window_height)

# identify_colors
def identify_colors(mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            print(f"{color_name} Contour Area:", cv2.contourArea(contour))

        largest_contour = max(contours, key=cv2.contourArea)  # 找最大輪廓
        largest_area = cv2.contourArea(largest_contour)  # 計算最大輪廓的面積
        if largest_area > 500:  # 過濾雜訊
            # print(f"{color_name} largest_area", largest_area)
            return color_name, largest_area
    return None, 0

# find_traffic_light
def find_traffic_light(image,c,conf):
    windownumber=f"traffic_light_{c+1}"

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅、黃、綠燈的 HSV 範圍
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    lower_yellow, upper_yellow = np.array([12, 100, 20]), np.array([35, 255, 255])
    lower_green, upper_green = np.array([50, 140, 90]), np.array([90, 255, 255])
    # lower_green = np.array([40, 100, 100])
    # upper_green = np.array([90, 255, 255])

    # 建立遮罩（包含兩個範圍）
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 判斷目前燈號
    light_color = "---"
    light_BGR = (255, 255, 255)
    largest_area = 0
    # 先嘗試識別不同燈號
    red_result = identify_colors(mask_red, "Red")
    yellow_result = identify_colors(mask_yellow, "Yellow")
    green_result = identify_colors(mask_green, "Green")

    if red_result[0]:  # 檢查是否識別到紅燈
        light_color = red_result[0]
        light_BGR = (0, 0, 255)
        largest_area = red_result[1]
    elif yellow_result[0]:  # 檢查是否識別到黃燈
        light_color = yellow_result[0]
        light_BGR = (0, 255, 255)
        largest_area = yellow_result[1]
    elif green_result[0]:  # 檢查是否識別到綠燈
        light_color = green_result[0]
        light_BGR = (0, 255, 0)
        largest_area = green_result[1]

    # 在上方添加黑色區域
    border_top = 40
    image = cv2.copyMakeBorder(image, border_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 顯示結果
    cv2.putText(image, f"{light_color}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_BGR, 2)
    cv2.putText(image, f"{conf}, {largest_area}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_BGR, 2)
    cv2.imshow(f"{windownumber}", image)
    
# yoloVideo
def video(image):
    result = model(image)
    cv2.imshow("Original Video",result[0].plot())


    traffic_light_class = 9  
    traffic_light_count = 0 
    conflist=[-1,-1,-1]
    boxlist=[-1,-1,-1]
    for box in result[0].boxes:
        cls = int(box.cls)  # 取得類別索引
        conf = float(box.conf)  # 取得該物件的信心值
        if cls == traffic_light_class and conf > 0.65:  # 只處理 traffic light
            if conf>conflist[0]:
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

            # 儲存圖片
            #save_path = f"output/traffic_light_{traffic_light_count}.jpg"
            #cv2.imwrite(save_path, cropped_img)
            #print((cropped_img))
        c+=1    
    print(f"總共擷取 {traffic_light_count} 個紅綠燈。")


# Startup
model = YOLO("YOLOv8l.pt")
target_classes = [0, 1, 2, 3, 5, 7, 9]  
# print(model.names)

# 開啟影片
video_path = "video/2025-01-19_11-47-10-front - Trim.mp4"
# video_path = "video/color_changed.mp4"
# video_path = "video/arrow_test.mp4"
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