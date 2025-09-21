import requests
import csv
import time

# 你需要在 https://openweathermap.org/api 注册获取 API KEY
API_KEY = 'e7441b7cdf76e4af17a8a38db45e88ce'

# 示例城市列表（可扩展为全球主要城市）
cities = [
    {'name': 'Beijing', 'lat': 39.9042, 'lon': 116.4074},
    {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737},
    {'name': 'Tokyo', 'lat': 35.6895, 'lon': 139.6917},
    {'name': 'Seoul', 'lat': 37.5665, 'lon': 126.9780},
    {'name': 'Bangkok', 'lat': 13.7563, 'lon': 100.5018},
    {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
    {'name': 'Dubai', 'lat': 25.2048, 'lon': 55.2708},
    {'name': 'Riyadh', 'lat': 24.7136, 'lon': 46.6753},
    {'name': 'Johannesburg', 'lat': -26.2041, 'lon': 28.0473},
    {'name': 'Nairobi', 'lat': -1.2921, 'lon': 36.8219},
    {'name': 'Cairo', 'lat': 30.0444, 'lon': 31.2357},
    {'name': 'Moscow', 'lat': 55.7558, 'lon': 37.6173},
    {'name': 'Istanbul', 'lat': 41.0082, 'lon': 28.9784},
    {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522},
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    {'name': 'Berlin', 'lat': 52.5200, 'lon': 13.4050},
    {'name': 'Madrid', 'lat': 40.4168, 'lon': -3.7038},
    {'name': 'Rome', 'lat': 41.9028, 'lon': 12.4964},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
    {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
    {'name': 'Mexico City', 'lat': 19.4326, 'lon': -99.1332},
    {'name': 'São Paulo', 'lat': -23.5505, 'lon': -46.6333},
    {'name': 'Buenos Aires', 'lat': -34.6037, 'lon': -58.3816},
    {'name': 'Lima', 'lat': -12.0464, 'lon': -77.0428},
    {'name': 'Toronto', 'lat': 43.651070, 'lon': -79.347015},
    {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
    {'name': 'Vancouver', 'lat': 49.2827, 'lon': -123.1207},
    {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093},
    {'name': 'Auckland', 'lat': -36.8485, 'lon': 174.7633},
]

# 采集时间点（可按需调整）
time_points = ['2025-09-18T12:00:00']  # 示例：只采集一个时间点

with open('city_weather.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['city', 'lat', 'lon', 'time', 'temp', 'humidity', 'wind', 'weather'])
    for city in cities:
        lat, lon = city['lat'], city['lon']
        url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
        try:
            resp = requests.get(url)
            data = resp.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            weather = data['weather'][0]['main']
            writer.writerow([
                city['name'], lat, lon, time_points[0], temp, humidity, wind, weather
            ])
            print(f"{city['name']} done.")
            time.sleep(1)  # 防止 API 速率限制
        except Exception as e:
            print(f"Error for {city['name']}: {e}")
