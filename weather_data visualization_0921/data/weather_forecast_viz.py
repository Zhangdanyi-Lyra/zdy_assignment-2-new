import pandas as pd
import pydeck as pdk
import streamlit as st
import os
from datetime import datetime

HOURLY_CSV = 'forecast_hourly.csv'
DAILY_CSV = 'forecast_daily.csv'

st.title('全球城市天气预测可视化 (One Call)')

if not os.path.exists(HOURLY_CSV):
    st.error('缺少 forecast_hourly.csv，请先运行 weather_onecall_fetch.py')
    st.stop()

@st.cache_data
def load_hourly():
    df = pd.read_csv(HOURLY_CSV)
    # 转换类型
    if 'dt' in df.columns:
        df['dt'] = pd.to_numeric(df['dt'], errors='coerce')
        df['time'] = pd.to_datetime(df['dt'], unit='s', utc=True)
    return df

df = load_hourly()

# 温度范围与颜色映射
min_t, max_t = float(df['temp'].min()), float(df['temp'].max())

@st.cache_data
def color_for_temp(t: float):
    # 线性插值蓝->青->绿->黄->红
    stops = [min_t, (min_t+max_t)/4, (min_t+max_t)/2, (3*min_t+max_t)/4, max_t]
    colors = [
        (0, 60, 255),
        (0, 200, 255),
        (0, 220, 120),
        (255, 230, 30),
        (255, 40, 0)
    ]
    if t <= stops[0]:
        return colors[0]
    if t >= stops[-1]:
        return colors[-1]
    for i in range(1, len(stops)):
        if t < stops[i]:
            t0, t1 = stops[i-1], stops[i]
            c0, c1 = colors[i-1], colors[i]
            ratio = (t - t0)/(t1 - t0 + 1e-9)
            return [int(c0[j] + (c1[j]-c0[j])*ratio) for j in range(3)]
    return colors[-1]

df['color'] = df['temp'].apply(color_for_temp)

# 风速映射半径（风速 m/s -> 像素缩放）
wind_col = 'wind_speed'
max_wind = max(1.0, float(df[wind_col].max()))
base_radius = st.sidebar.slider('基础半径 (m)', 15000, 80000, 30000, 5000)
scale = st.sidebar.slider('风速放大系数', 1.0, 6.0, 2.5, 0.5)
df['radius'] = base_radius * (df[wind_col] / max_wind * scale + 0.2)

# 时间筛选
all_times = sorted(df['time'].unique())
idx = st.sidebar.slider('时间步', 0, len(all_times)-1, 0)
current_time = all_times[idx]
frame_df = df[df['time'] == current_time]

st.caption(f"当前时间: {pd.to_datetime(current_time).isoformat()} | 样本: {len(frame_df)} | 温度范围: {min_t:.1f}~{max_t:.1f}°C")

layer = pdk.Layer(
    'ScatterplotLayer',
    data=frame_df,
    get_position='[lon, lat]',
    get_fill_color='color',
    get_radius='radius',
    pickable=True,
    auto_highlight=True,
    radius_min_pixels=3,
)

# 轨迹动画（可选：最近若干小时残影）
trail_hours = st.sidebar.slider('残影小时数', 0, 12, 0)
trail_layers = []
if trail_hours > 0:
    trail_cut = pd.to_datetime(current_time) - pd.Timedelta(hours=trail_hours)
    trail_df = df[(df['time'] < current_time) & (df['time'] >= trail_cut)]
    if not trail_df.empty:
        # 降低透明度
        trail_df = trail_df.copy()
        trail_df['trail_color'] = trail_df['color'].apply(lambda c: [int(c[0]*0.4), int(c[1]*0.4), int(c[2]*0.4)])
        trail_layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=trail_df,
                get_position='[lon, lat]',
                get_fill_color='trail_color',
                get_radius='radius',
                pickable=False,
                radius_min_pixels=2,
            )
        )

view_state = pdk.ViewState(
    latitude=float(frame_df['lat'].mean()),
    longitude=float(frame_df['lon'].mean()),
    zoom=1.5,
    pitch=30,
)

r = pdk.Deck(
    layers=trail_layers + [layer],
    initial_view_state=view_state,
    tooltip={'text': '{city}\n温度: {temp}°C\n体感: {feels_like}°C\n风速: {wind_speed}m/s\n云量: {clouds}%\n天气: {weather_main}'}
)

st.pydeck_chart(r)

# 自动播放
play = st.sidebar.checkbox('自动播放')
if play:
    import time as _t
    _t.sleep(0.6)
    next_idx = (idx + 1) % len(all_times)
    st.experimental_set_query_params(t=next_idx)
    st.experimental_rerun()
