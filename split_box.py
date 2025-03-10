'''
從遮罩影像 (mask) 提取所有物件輪廓並存檔
'''
import cv2
import numpy as np
import os

# 讀取影像
# image = cv2.imread(r"output\traffic_light_1.jpg")
image = cv2.imread(r"pic\arrow.png")
# image = cv2.imread(r"pic\green.png")
# image = cv2.imread(r"pic\red.jpg")
# image = cv2.imread(r"pic\yello.jpg")

# 轉換為 HSV 色彩空間
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義紅、黃、綠燈的 HSV 範圍
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
lower_yellow, upper_yellow = np.array([12, 100, 20]), np.array([35, 255, 255])
lower_green, upper_green = np.array([50, 140, 90]), np.array([90, 255, 255])

# 建立紅燈遮罩（包含兩個範圍）
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# 確保輸出資料夾存在
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

split_traffic_light_ = 0

def split_box(image, mask, color_name):
    global split_traffic_light_
    
    # 找出遮罩中的輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > 50:  # 過濾掉小雜訊
            x, y, w, h = cv2.boundingRect(contour)  # 計算外接矩形
            cropped_img = image[y:y+h, x:x+w]  # 裁剪出物件

            # 儲存影像
            split_traffic_light_ += 1
            save_path = os.path.join(output_dir, f"splited_{color_name}_{split_traffic_light_}.jpg")
            cv2.imwrite(save_path, cropped_img)
            print(f"已存檔: {save_path}")

# 執行影像分割並存檔
split_box(image, mask_red, "red")
split_box(image, mask_yellow, "yellow")
split_box(image, mask_green, "green")

# 顯示影像與遮罩
cv2.imshow("Original Image", image)
cv2.imshow("Red Mask", mask_red)
cv2.imshow("Yellow Mask", mask_yellow)
cv2.imshow("Green Mask", mask_green)

cv2.waitKey(0)
cv2.destroyAllWindows()
