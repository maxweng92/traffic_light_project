import googlemaps
import json
import os.path

class GoogleAPIClient:
    # 此類並不需要處理 Directions API 的部分，只要提供 API 金鑰
    API_KEY = 'AIzaSyBzO5mojGHgvT5YvFA6OK8YWj1Pk54YXeM'

    def __init__(self) -> None:
        # 使用 API 金鑰初始化 googlemaps 客戶端
        self.gmaps = googlemaps.Client(key=self.API_KEY)

    def get_directions(self, origin, destination, mode="driving"):
        # 使用 Directions API 查詢路線
        directions_result = self.gmaps.directions(origin, 
                                                  destination, 
                                                  mode=mode,  # 交通方式: driving, walking, bicycling, transit
                                                  departure_time="now")
        return directions_result

# 設定起點與終點
origin = '25.024650611596684, 121.54479088527705'  # 起點的緯度和經度
destination = '25.046246094858525, 121.51738404083065'  # 終點的緯度和經度

# 初始化 GoogleAPIClient 並查詢路線
client = GoogleAPIClient()
directions_result = client.get_directions(origin, destination)

# 將結果保存為 JSON 文件
with open('directions_result_driving.json', 'w', encoding='utf-8') as json_file:
    json.dump(directions_result, json_file, ensure_ascii=False, indent=4)

print("結果已保存為 directions_result.json")
