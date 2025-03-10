import cv2
import numpy as np

# 讀取影像
image = cv2.imread(r"pic\2025-01-19_11-54-11-front - Trim - frame at 0m1s.jpg")
# 目標視窗的寬度與高度（可以自訂）
target_width = 1280
target_height = 720
# 設定視窗名稱和可調整大小的屬性
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# 調整視窗大小
cv2.resizeWindow("frame", target_width, target_height)


# 將影像從BGR色彩空間轉換到HSV色彩空間
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義紅色的HSV範圍
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
# 定義綠色的HSV範圍
lower_green = np.array([35, 70, 50])   # 綠色的下限
upper_green = np.array([85, 255, 255]) # 綠色的上限


# 創建紅色區域的遮罩
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)
# 創建綠色區域的遮罩
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)


# 對原始影像應用遮罩，只保留紅色區域
red_image = cv2.bitwise_and(image, image, mask=red_mask)

cv2.imshow("frame", red_image)
cv2.imwrite("output1.jpg", red_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
cv2.waitKey(0)

# 檢測綠色區域的輪廓
# contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)  # 獲取邊界框
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 繪製綠色邊界框

# 尋找紅色區域的輪廓
contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 繪製紅色區域的輪廓
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)



# 顯示影像
cv2.imshow("frame", image)
cv2.imwrite("output2.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])

# 等待按鍵按下
cv2.waitKey(0)

# 關閉所有視窗
cv2.destroyAllWindows()