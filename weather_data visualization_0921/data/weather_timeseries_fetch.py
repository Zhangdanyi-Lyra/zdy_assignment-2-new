import os
import csv
import time
import math
import json
import queue
import threading
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests

API_KEY = os.getenv('OWM_API_KEY', 'e7441b7cdf76e4af17a8a38db45e88ce')  # 可用环境变量覆盖
OUTPUT_CSV = 'city_weather_timeseries.csv'
LOG_FILE = 'weather_fetch.log'
INTERVAL_SEC = 900  # 每次采集间隔（秒）: 15 分钟，可调整
MAX_RUNS = 4        # 运行轮数；设为 None 表示无限循环
CONCURRENCY = 6     # 并发线程数
RETRY = 3           # 单请求重试次数
BACKOFF = 2         # 退避基数
TIMEOUT = 10        # HTTP 超时
MAX_HOURS_KEEP = 72  # 仅保留最近多少小时数据
PID_FILE = 'weather_timeseries.pid'

# 30 城市列表（与基础脚本一致）
CITIES = [
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

CSV_HEADER = [
    'timestamp_iso', 'timestamp_unix', 'city', 'lat', 'lon',
    'temp', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'clouds', 'weather_main', 'weather_desc'
]

os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

session = requests.Session()


def fetch_city(city: Dict[str, Any]) -> Dict[str, Any]:
    lat = city['lat']
    lon = city['lon']
    url = (
        f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}'
        f'&appid={API_KEY}&units=metric'
    )
    last_exc = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(f'status={resp.status_code} body={resp.text[:120]}')
            data = resp.json()
            main = data.get('main', {})
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            weather = (data.get('weather') or [{}])[0]
            dt_unix = data.get('dt', int(time.time()))
            dt_iso = datetime.fromtimestamp(dt_unix, tz=timezone.utc).isoformat()
            return {
                'timestamp_iso': dt_iso,
                'timestamp_unix': dt_unix,
                'city': city['name'],
                'lat': lat,
                'lon': lon,
                'temp': main.get('temp'),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'wind_speed': wind.get('speed'),
                'wind_deg': wind.get('deg'),
                'clouds': clouds.get('all'),
                'weather_main': weather.get('main'),
                'weather_desc': weather.get('description'),
            }
        except Exception as e:
            last_exc = e
            wait = BACKOFF ** (attempt - 1)
            logging.warning(f"{city['name']} attempt {attempt} failed: {e}; retry in {wait}s")
            time.sleep(wait)
    logging.error(f"{city['name']} all retries failed: {last_exc}")
    return {}


def worker(q: 'queue.Queue[Dict[str, Any]]', results: List[Dict[str, Any]]):
    while True:
        try:
            city = q.get_nowait()
        except queue.Empty:
            return
        data = fetch_city(city)
        if data:
            results.append(data)
        q.task_done()


def write_rows(rows: List[Dict[str, Any]]):
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def dedupe_and_trim():
    if not os.path.isfile(OUTPUT_CSV):
        return
    try:
        import pandas as pd  # 惰性导入，避免未安装时报错
    except Exception:
        logging.warning('pandas 不可用，跳过去重与截断。')
        return
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if df.empty:
            return
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp_unix', 'city'], keep='last')
        # 截断
        latest_ts = df['timestamp_unix'].max()
        cutoff = latest_ts - MAX_HOURS_KEEP * 3600
        df = df[df['timestamp_unix'] >= cutoff]
        df = df.sort_values(['timestamp_unix', 'city'])
        df.to_csv(OUTPUT_CSV, index=False)
        after = len(df)
        logging.info(f'dedupe/trim: rows {before}->{after}, cutoff {cutoff}')
    except Exception as e:
        logging.error(f'dedupe/trim 失败: {e}')


def run_once(run_idx: int):
    q: 'queue.Queue[Dict[str, Any]]' = queue.Queue()
    for c in CITIES:
        q.put(c)
    threads = []
    results: List[Dict[str, Any]] = []
    for _ in range(CONCURRENCY):
        t = threading.Thread(target=worker, args=(q, results), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if results:
        write_rows(results)
        logging.info(f"Run {run_idx}: wrote {len(results)} rows")
    else:
        logging.warning(f"Run {run_idx}: no data collected")


def main():
    logging.info('=== Weather timeseries fetch started ===')
    # 写入 PID 文件
    try:
        with open(PID_FILE, 'w') as pf:
            pf.write(str(os.getpid()))
    except Exception as e:
        logging.warning(f'写 PID 文件失败: {e}')
    run_idx = 1
    while True:
        start = time.time()
        run_once(run_idx)
        # 写完一轮后进行去重与截断
        dedupe_and_trim()
        run_idx += 1
        if MAX_RUNS and run_idx > MAX_RUNS:
            break
        elapsed = time.time() - start
        sleep_for = max(0, INTERVAL_SEC - elapsed)
        logging.info(f"Sleeping {sleep_for:.1f}s before next run...")
        time.sleep(sleep_for)
    logging.info('=== Completed ===')
    # 结束时移除 PID 文件
    try:
        if os.path.isfile(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


if __name__ == '__main__':
    main()
