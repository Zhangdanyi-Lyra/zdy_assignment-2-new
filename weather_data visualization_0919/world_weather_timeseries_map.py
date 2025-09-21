import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

INPUT = 'city_weather.csv'  # now includes wind_deg
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / 'world_weather_30d.html'

# 读数据
if not Path(INPUT).exists():
    raise SystemExit(f'Missing {INPUT}.')

df = pd.read_csv(INPUT)
# 兼容风向列
if 'wind_deg' not in df.columns:
    df['wind_deg'] = np.nan

# 规范时间并仅保留近30天
df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
df = df.dropna(subset=['time'])
max_time = df['time'].max()
min_time = max_time - pd.Timedelta(days=30)
df = df[(df['time'] >= min_time) & (df['time'] <= max_time)].copy()

# 日均聚合
df['date'] = df['time'].dt.floor('D')
agg = df.groupby(['city','lat','lon','date'], as_index=False).agg(
    temp=('temp','mean'),
    humidity=('humidity','mean'),
    wind=('wind','mean'),
    wind_deg=('wind_deg','mean')
)
agg['date_str'] = agg['date'].dt.strftime('%Y-%m-%d')

# 温度色标范围
t_min, t_max = float(agg['temp'].min()), float(agg['temp'].max())
if not np.isfinite(t_min):
    t_min = 0.0
if not np.isfinite(t_max):
    t_max = 40.0

# 箭头终点计算（风速缩放，风向为来向）
R = 111.0  # km per degree
scale_km = 2.0
rad = np.deg2rad(agg['wind_deg'].fillna(0.0))
ux = np.sin(rad)
uy = np.cos(rad)
length_deg = (agg['wind'] * scale_km / R).fillna(0.0)
agg['lon2'] = agg['lon'] + ux * length_deg
agg['lat2'] = agg['lat'] + uy * length_deg

MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN', '')

frames = []
for d, g in agg.groupby('date_str'):
    # 湿度底图（有 token 用 densitymapbox，没 token 用透明散点）
    if MAPBOX_TOKEN:
        humidity_layer = go.Densitymapbox(
            lat=g['lat'], lon=g['lon'], z=g['humidity'],
            radius=40, colorscale='Blues', opacity=0.35,
            name='Humidity', showscale=False,
        )
    else:
        humidity_layer = go.Scattergeo(
            lat=g['lat'], lon=g['lon'],
            marker=dict(
                size=(g['humidity']/max(1.0, g['humidity'].max())*18+6),
                color=g['humidity'], colorscale='Blues', opacity=0.25,
            ),
            mode='markers', name='Humidity (points)',
        )

    temp_points = go.Scattergeo(
        lat=g['lat'], lon=g['lon'],
        text=g['city'],
        marker=dict(
            size=(g['wind']/max(1.0, g['wind'].max())*14+6),
            color=g['temp'], colorscale=[[0,'#003CFF'], [0.25,'#00C8FF'], [0.5,'#00DC78'], [0.75,'#FFE61E'], [1,'#FF2800']],
            cmin=t_min, cmax=t_max,
            line=dict(width=0)
        ),
        hovertemplate='City: %{text}<br>Temp: %{marker.color:.1f}°C<br>Hum: %{customdata[0]:.0f}%<br>Wind: %{customdata[1]:.1f} m/s',
        customdata=np.c_[g['humidity'].values, g['wind'].values],
        mode='markers', name='Temperature'
    )

    arrow = go.Scattergeo(
        lon=np.r_[g['lon'].values, None, g['lon2'].values],
        lat=np.r_[g['lat'].values, None, g['lat2'].values],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.7)', width=1),
        name='Wind Direction',
    )

    frames.append(go.Frame(data=[humidity_layer, temp_points, arrow], name=d))

first = frames[0].data if frames else []
fig = go.Figure(data=first, frames=frames)
fig.update_layout(
    title='Global Cities Weather — Last 30 Days (Daily Avg)\nHumidity layer + Wind arrows',
    geo=dict(
        projection_type='natural earth',
        showcountries=True, showcoastlines=True, showland=True,
        landcolor='#111', bgcolor='#000', lakecolor='#000'
    ),
    paper_bgcolor='#000', plot_bgcolor='#000',
    font=dict(color='#EEE'),
    updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ],
        'x': 0.02, 'y': 0.02, 'xanchor': 'left', 'yanchor': 'bottom'
    }]
)

if MAPBOX_TOKEN:
    fig.update_layout(mapbox_accesstoken=MAPBOX_TOKEN)

fig.write_html(str(OUTPUT_HTML), include_plotlyjs='cdn', auto_play=False)
print(f'Wrote {OUTPUT_HTML}')
