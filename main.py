import cv2
from ultralytics import YOLO
import numpy as np

window_width = 800
window_height = 600
cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Video", window_width, window_height)

#利用條件，充分判斷綠燈的狀態
def green_arrow_decision(green_result,green_mask_result, green_assume_left, green_assume_right):
    #green_decision_list
    g = [green_result[0], green_mask_result[0], green_assume_left[0], green_assume_right[0]] 
    print("綠色陣列:",g)
    if (g[2] == "^"):
        return "<"
    elif (g[3] == "^"):
        return ">"
    elif ((g[0] == "^" and g[1] == "^") or (g[0] == "^" and g[2] == ">" and g[3] == "<")):
        return "^"
    elif (g[0] == "<" and g[1] == "<"):
        return "<"
    elif (g[0] == ">" and g[1] == ">"):
        return ">"
    else:
        return "can not decide"

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
        # print("Aspect_ratio:", aspect_ratio)
        if 0.8 <= aspect_ratio <= 1.3:  # 接近正方形，原先是0.8~1.3
            if color_name == "Red" or color_name == "Yellow":
            # match color_name:
                # case "Red" | "Yellow": 
                if largest_area > 500:  # 過濾雜訊
                    return color_name, largest_area
            # 加入了green的mask
            elif color_name.startswith("Green"):
                # case "Green":
                if largest_area > 150:  # 過濾雜訊，原先是150
                    # [image, mask]
                    for __ in range(2):
                        if color_name.startswith("Green_mask"):
                            t = 1  # mask
                        else:
                            t = 0  # image
                        # make rotate if assume left or right
                        if color_name == "Green_assume_left":
                            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        elif color_name == "Green_assume_right":
                            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        if t == 0:
                            results = model5(image) #image to mask> test gray
                        else:
                            results = model5_gray(mask)

                        ArrowSequence = ["<", "^", ">"] #表示三種class的方向
                        for result in results:
                            if hasattr(result, 'probs') and result.probs is not None:
                                # 取得分類機率最大者
                                # predicted_class = result.probs.top1
                                # 获取所有类别的置信度
                                confidences = [round(prob.item(), 4) for prob in result.probs.data]  # 保留4位小数
                                predicted_class = result.probs.top1  # 最高概率的类别索引
                                max_confidence = confidences[predicted_class]  # 最高置信度
                                ans = ArrowSequence[predicted_class]  # 对应的类别名称

                                # 打印所有置信度
                                print("All confidences:")
                                for idx, conf in enumerate(confidences):
                                    class_name = ArrowSequence[idx]  # 假设 ArrowSequence 是类别名称列表
                                    print(f"  {class_name}: {conf}")

                                # print(f"Predicted Class: {ans}, Confidence: {max_confidence}")
                                # 
                                confidence = round(result.probs.data[predicted_class].item(), 2)  # 直接保留兩位小數
                                ans = ArrowSequence[predicted_class]
                                #print image or mask
                                print(color_name)

                                print(f"Predicted Class: {ans}, Confidence: {confidence}")
                                #edit. for green result, and try return result forever
                                #cv2.imshow("Green Pre", mask)
                                #cv2.waitKey(0) # not wait key for 4 result 
                                #cv2.destroyWindow("Green Pre")
                                #if confidence > 0.5:  # Only return if confidence > 0.5
                                return ans, confidence
                            else:
                                print("No classification results found.")
                                return "X", 0.0  # Assuming confidence should be 0.0 when no results
                        # # cv2.imshow("Green Pre",  model5.predict(image))
                        # result = model5(image)
                        # result[0].show()
                        # cv2.imshow("Green Pre", mask)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("Green Pre")
                        # cv2.imshow("Green Pre", mask)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("Green Pre")
                    
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

    # 用集合來儲存紅綠燈的顏色，這樣可以避免重複，或對調
    light_colors = set([])
    def Display_light_colors():
        light_colors_sorted_list=('Red', 'Yellow', 'Green', '<', '^', '>')
        print_list=[]
        for i in light_colors_sorted_list:
            if i in light_colors:
                print_list.append(i)
        return print_list


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
        green_mask_result = identify_colors(mask_green,frame, "Green_mask")
        green_assume_left = identify_colors(mask_green,frame, "Green_assume_left")
        green_assume_right = identify_colors(mask_green,frame, "Green_assume_right")
        green_arrow_result = green_arrow_decision(green_result,green_mask_result, green_assume_left, green_assume_right)
        # 判斷目前燈號
        light_BGR = (255, 255, 255)
        largest_area = 0

        if red_result[0]:  # 檢查是否識別到紅燈
            light_colors.add(red_result[0])
            light_BGR = (0, 0, 255)
            largest_area = red_result[1]
        elif yellow_result[0]:  # 檢查是否識別到黃燈
            light_colors.add(yellow_result[0])
            light_BGR = (0, 255, 255)
            largest_area = yellow_result[1]
        elif green_result[0]:  # 檢查是否識別到綠燈
            if(green_arrow_result != "can not decide"):
                light_colors.add(green_arrow_result)
            light_BGR = (0, 255, 0)
            largest_area = green_result[1]
            # cv2.putText(mask_green, f"{c}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_BGR, 2)
            # cv2.imshow("Green Mask", mask_green)
            # cv2.waitKey(0)

        print(f"目前的紅綠燈狀態：{Display_light_colors()}")

        print("綠色燈結果:", green_arrow_result)
        #display green arrow
        cv2.imshow("Green Pre", mask_green)
        cv2.waitKey(0)
        cv2.destroyWindow("Green Pre")
        #if green_arrow_result != "can not decide":
        #    time.sleep(1)

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
    #image = image_copy，如果要執行Display_light_colors，放在for的外面
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
model5_gray = YOLO("bestSingleArrowTrain22_MixedGray.pt")
model5 = YOLO("bestSingleArrowTrain30.pt")
#model5 = YOLO("bestSingleArrowTrain14.pt")
target_classes = [0, 1, 2, 3, 5, 7, 9]  
# print(model.names)

# 定義紅、黃、綠燈的 HSV 範圍
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
lower_yellow, upper_yellow = np.array([12, 100, 20]), np.array([35, 255, 255])
lower_green, upper_green = np.array([65, 155, 90]), np.array([120, 255, 255])
#lower_green, upper_green = np.array([65, 155, 90]), np.array([90, 255, 255])
# lower_green = np.array([40, 100, 100])
# upper_green = np.array([90, 255, 255])

# 開啟影片
#video_path = "video/2025-01-19_11-47-10-front - Trim.mp4"
#video_path = "video/color_changed.mp4"
#video_path = "video/rotated_cropped_video.mp4"
#video_path = "video/2025-01-29_11-16-31-front_TRIM.mp4"
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
