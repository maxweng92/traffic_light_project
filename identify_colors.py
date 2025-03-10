import cv2
import numpy as np

# 讀取影像
# image = cv2.imread(r"pic\red.jpg")
# image = cv2.imread(r"pic\green.png")
# image = cv2.imread(r"pic\yello.jpg")
# image = cv2.imread(r"pic\arrow.png")
image = cv2.imread(r"output\splited_green_1.jpg")


# 轉換為 HSV 色彩空間
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 紅燈範圍（包含兩個範圍）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 黃燈範圍
lower_yellow = np.array([12, 100, 20])
upper_yellow = np.array([35, 255, 255])

# 綠燈範圍
# lower_green = np.array([40, 100, 100])
# upper_green = np.array([90, 255, 255])
lower_green = np.array([50, 140, 90])
upper_green = np.array([90, 255, 255])

# 建立紅色遮罩（兩個範圍）
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

# 建立黃色遮罩
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# 建立綠色遮罩
mask_green = cv2.inRange(hsv, lower_green, upper_green)

def identify_colors(mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            print(color_name+"Contour Area:", cv2.contourArea(contour))

        largest_contour = max(contours, key=cv2.contourArea)  # 找最大輪廓
        if cv2.contourArea(largest_contour) > 500:  # 過濾雜訊
            return color_name
    return None

# 判斷目前燈號
light_color = None
light_BGR = (255, 255, 255)
if identify_colors(mask_red, "紅燈"):
    light_color = "red"
    light_BGR = (0,0,255)
elif identify_colors(mask_yellow, "黃燈"):
    light_color = "yello"
    light_BGR = (0,255,255)
elif identify_colors(mask_green, "綠燈"):
    light_color = "green"
    light_BGR = (0,255,0)
print(f"目前的紅綠燈狀態：{light_color}")

# 繪製燈號框
if light_color:
    cv2.putText(image, f"{light_color}", (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, light_BGR, 2)

# 顯示原始影像
cv2.imshow("Original Image", image)

# 顯示紅色遮罩
cv2.imshow("Red Mask", mask_red)

# 顯示黃色遮罩
cv2.imshow("Yellow Mask", mask_yellow)

# 顯示綠色遮罩
cv2.imshow("Green Mask", mask_green)

cv2.waitKey(0)
cv2.destroyAllWindows()