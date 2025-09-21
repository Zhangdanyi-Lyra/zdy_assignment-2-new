import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import argparse

def parse_args():
    p = argparse.ArgumentParser(description='Global weather map with humidity base + wind arrows')
    p.add_argument('--input', default='city_weather.csv', help='CSV with wind_deg')
    p.add_argument('--out', default='outputs/world_weather_30d_wind_humidity.html', help='Output HTML path')
    p.add_argument('--granularity', choices=['daily','hourly'], default='daily', help='Frame granularity')
    p.add_argument('--hour-step', type=int, default=1, help='Subsample hours (hourly only)')
    p.add_argument('--arrow-scale-km', type=float, default=2.0, help='Arrow length scale (km per m/s)')
    p.add_argument('--arrow-head-deg', type=float, default=25.0, help='Arrow head half-angle (degrees)')
    p.add_argument('--arrow-head-scale', type=float, default=0.5, help='Arrow head length as fraction of shaft')
    p.add_argument('--arrow-width-scale', type=float, default=1.0, help='Global scale for wind arrow line widths')
    p.add_argument('--fps', type=int, default=3, help='Playback speed (approx frames per second)')
    # Temperature color range
    p.add_argument('--tmin', type=float, default=None, help='Fixed temperature min for color scale (°C)')
    p.add_argument('--tmax', type=float, default=None, help='Fixed temperature max for color scale (°C)')
    # Glow mapping resolution (continuous-like)
    p.add_argument('--glow-bins', type=int, default=12, help='Number of wind bins for glow (higher ≈ more continuous)')
    # Rays (radiating curves) options
    p.add_argument('--rays', type=int, default=16, help='Number of radiating curves per city (doubled count)')
    p.add_argument('--rays-mode', choices=['radial','three'], default='radial', help='Ray strategy: full radial or exactly three per city')
    p.add_argument('--ray-length-km', type=float, default=300.0, help='Ray length in km (base, scaled by wind)')
    p.add_argument('--ray-curvature-km', type=float, default=6.0, help='Ray curvature amplitude in km（强烈曲线风格推荐 6.0）')
    p.add_argument('--ray-angle-deg', type=float, default=20.0, help='Half-angle offset for side rays in three mode')
    p.add_argument('--ray-segments', type=int, default=8, help='Polyline segments per ray (smoothness, 建议8用于平滑强曲线)')
    p.add_argument('--ray-base-width', type=float, default=1.0, help='Inner ray width (thin by 2 units approx)')
    p.add_argument('--no-rays', action='store_true', help='Disable radiating curves')
    p.add_argument('--rays-step', type=int, default=1, help='Draw every k-th ray only (decimate)')
    p.add_argument('--rays-rotate', action='store_true', help='Rotate which rays are drawn across frames')
    p.add_argument('--rays-topk', type=int, default=None, help='Only draw rays for top-K windy cities per frame')
    # KNN length options
    p.add_argument('--ray-length-mode', choices=['wind','knn'], default='wind', help='How to set ray length')
    p.add_argument('--ray-length-mult', type=float, default=3.0, help='Multiply computed ray length (after mode, 建议3.0用于加长射线)')
    p.add_argument('--ray-neighbor-k', type=int, default=2, help='k-th nearest neighbor distance used for length')
    p.add_argument('--ray-neighbor-frac', type=float, default=1.05, help='Multiply kNN distance to set length')
    p.add_argument('--ray-length-min-km', type=float, default=5.0, help='Minimum ray length (km)')
    p.add_argument('--ray-length-max-km', type=float, default=2500.0, help='Maximum ray length (km)')
    # Ray width scaling
    p.add_argument('--ray-width-scale', type=float, default=0.3333, help='Scale factor for ray line widths')
    # Tail blinking dots
    p.add_argument('--tail-blink', action='store_true', default=True, help='Add blinking circles at ray tails')
    p.add_argument('--tail-size', type=float, default=6.0, help='Base size for tail circles')
    p.add_argument('--tail-size-scale', type=float, default=1.0, help='Multiply tail size for quick global shrink/grow')
    p.add_argument('--tail-opacity', type=float, default=0.9, help='Base opacity for tail circles (0-1)')
    p.add_argument('--tail-omega', type=float, default=2.0, help='Blink angular speed (radians per frame)')
    # Emission effect (grow from center outward)
    p.add_argument('--ray-emit', action='store_true', help='Animate rays to emit from center outward')
    p.add_argument('--ray-emit-period', type=int, default=16, help='Frames per emission cycle')
    p.add_argument('--ray-emit-phase-jitter', type=float, default=0.3, help='Randomized phase per city (0-1 of cycle)')
    p.add_argument('--ray-emit-min-frac', type=float, default=0.06, help='Minimum shown fraction to stay visible')
    # Jellyfish-like undulation along rays (tail sway)
    p.add_argument('--ray-undulate', action='store_true', help='Enable jellyfish-like undulating motion on rays')
    p.add_argument('--ray-undulate-amp-km', type=float, default=0.35, help='Undulation amplitude at tip in km (orthogonal offset)')
    p.add_argument('--ray-undulate-waves', type=float, default=1.5, help='Number of waves along the ray length')
    p.add_argument('--ray-undulate-period', type=float, default=18.0, help='Frames per undulation cycle (lower=faster)')
    p.add_argument('--ray-undulate-decay', type=float, default=1.2, help='Amplitude scales ~ s^decay from base to tip (s in [0,1])')
    # Firework burst effect (optional overlay)
    p.add_argument('--firework', action='store_true', help='Enable firework-like burst particles from city centers')
    p.add_argument('--fw-sparks', type=int, default=18, help='Number of sparks per city per burst')
    p.add_argument('--fw-burst-period', type=int, default=24, help='Frames per burst cycle')
    p.add_argument('--fw-burst-jitter', type=float, default=0.4, help='Random phase jitter per city (0-1)')
    p.add_argument('--fw-expansion-km', type=float, default=220.0, help='Max expansion radius in km')
    p.add_argument('--fw-fade-power', type=float, default=2.2, help='Spark opacity falloff power with progress')
    p.add_argument('--fw-spark-size', type=float, default=6.0, help='Base size of sparks')
    # Firework trails and core flash
    p.add_argument('--fw-trail', action='store_true', help='Enable spark trails (fading points along the path)')
    p.add_argument('--fw-trail-samples', type=int, default=5, help='Number of samples per spark trail')
    p.add_argument('--fw-trail-span', type=float, default=0.35, help='Trail span as fraction of cycle behind current progress')
    p.add_argument('--fw-trail-fade', type=float, default=1.5, help='Additional fade power for trails')
    p.add_argument('--fw-trail-size-scale', type=float, default=0.7, help='Relative size of trail points vs spark size')
    p.add_argument('--fw-core', action='store_true', help='Enable bright core flash at burst center')
    p.add_argument('--fw-core-duration', type=float, default=0.18, help='Core flash duration fraction of cycle (0-1)')
    p.add_argument('--fw-core-size', type=float, default=10.0, help='Core flash size')
    p.add_argument('--fw-core-alpha', type=float, default=0.95, help='Max alpha of core flash')
    # Background music (removed)
    # Color binning for ray gradients
    p.add_argument('--ray-color-bins', type=int, default=8, help='Number of temperature bins for ray coloring gradients（建议8或更高以获得平滑渐变）')
    # One-shot lightweight preset (applies after parse)
    p.add_argument('--lite', action='store_true', help='Enable lightweight defaults for smoother playback')
    return p.parse_args()

args = parse_args()

# Apply lightweight preset if requested (override only if not explicitly set via flags)
if args.lite:
    # Increase hour step if hourly by default
    if args.granularity == 'hourly' and args.hour_step == 1:
        args.hour_step = 6
    # Moderate glow bins
    if args.glow_bins == 12:
        args.glow_bins = 6
    # Rays preset
    if args.rays == 16:
        args.rays = 12
    if args.ray_segments == 12:
        args.ray_segments = 8
    if args.ray_base_width == 1.0:
        args.ray_base_width = 0.7
    if args.rays_step == 1:
        args.rays_step = 3
    if not args.rays_rotate:
        args.rays_rotate = True
    if args.rays_topk is None:
        args.rays_topk = 18

INPUT = args.input
OUTPUT_HTML = Path(args.out)
OUTPUT_HTML.parent.mkdir(exist_ok=True)

if not Path(INPUT).exists():
    raise SystemExit(f'Missing {INPUT}. Run build_city_weather_30d.py first.')

df = pd.read_csv(INPUT)
if 'wind_deg' not in df.columns:
    df['wind_deg'] = np.nan

df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
df = df.dropna(subset=['time'])
max_time = df['time'].max()
min_time = max_time - pd.Timedelta(days=30)
df = df[(df['time'] >= min_time) & (df['time'] <= max_time)].copy()

if args.granularity == 'daily':
    df['frame'] = df['time'].dt.floor('D')
else:
    df['frame'] = df['time'].dt.floor('h')
    if args.hour_step > 1:
        sel = (df['frame'].dt.hour % args.hour_step) == 0
        df = df[sel].copy()

agg = df.groupby(['city','lat','lon','frame'], as_index=False).agg(
    temp=('temp','mean'),
    humidity=('humidity','mean'),
    wind=('wind','mean'),
    wind_deg=('wind_deg','mean'),
    weather=('weather', lambda x: x.mode().iloc[0] if not x.mode().empty else '')
)
agg['frame_str'] = agg['frame'].dt.strftime('%Y-%m-%d' if args.granularity=='daily' else '%Y-%m-%d %H:%M')

R = 111.0
scale_km = args.arrow_scale_km
rad = np.deg2rad(agg['wind_deg'].fillna(0.0))
ux = np.sin(rad)
uy = np.cos(rad)
length_deg = (agg['wind'] * scale_km / R).fillna(0.0)
agg['lon2'] = agg['lon'] + ux * length_deg
agg['lat2'] = agg['lat'] + uy * length_deg

# Arrow heads
head_angle = np.deg2rad(args.arrow_head_deg)
head_frac = args.arrow_head_scale
hx1 = np.sin(rad + head_angle)
hy1 = np.cos(rad + head_angle)
hx2 = np.sin(rad - head_angle)
hy2 = np.cos(rad - head_angle)
head_len = length_deg * head_frac
agg['lonh1'] = agg['lon2'] - hx1 * head_len
agg['lath1'] = agg['lat2'] - hy1 * head_len
agg['lonh2'] = agg['lon2'] - hx2 * head_len
agg['lath2'] = agg['lat2'] - hy2 * head_len

t_min, t_max = float(agg['temp'].min()), float(agg['temp'].max())
if not np.isfinite(t_min): t_min = 0.0
if not np.isfinite(t_max): t_max = 40.0
# Override with user-provided fixed range
if args.tmin is not None:
    t_min = args.tmin
if args.tmax is not None:
    t_max = args.tmax

MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN', '')

# --- Color utilities for temperature-based ray coloring ---
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_rgb(c1, c2, t):
    return (
        int(round(_lerp(c1[0], c2[0], t))),
        int(round(_lerp(c1[1], c2[1], t))),
        int(round(_lerp(c1[2], c2[2], t)))
    )

def _rgb_to_rgba_str(rgb, a):
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{max(0.0,min(1.0,float(a))):.3f})'

def _rgb_to_hex(rgb):
    return f'#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}'

_palette_points = [
    (0.00, _hex_to_rgb('003CFF')),
    (0.25, _hex_to_rgb('00C8FF')),
    (0.50, _hex_to_rgb('00DC78')),
    (0.75, _hex_to_rgb('FFE61E')),
    (1.00, _hex_to_rgb('FF2800')),
]

def temp_to_rgb(val, tmin, tmax):
    if not np.isfinite(val):
        val = (tmin + tmax) * 0.5
    z = (val - tmin) / (max(1e-9, (tmax - tmin)))
    z = float(np.clip(z, 0.0, 1.0))
    for i in range(len(_palette_points)-1):
        x0, c0 = _palette_points[i]
        x1, c1 = _palette_points[i+1]
        if z >= x0 and z <= x1:
            tt = (z - x0) / (x1 - x0 + 1e-9)
            return _lerp_rgb(c0, c1, tt)
    return _palette_points[-1][1]

def lighten_rgb(rgb, frac):
    return (
        int(round(_lerp(rgb[0], 255, frac))),
        int(round(_lerp(rgb[1], 255, frac))),
        int(round(_lerp(rgb[2], 255, frac)))
    )

# --- Descriptive annotations (English) ---
def deg_to_cardinal(deg):
    if not np.isfinite(deg):
        return 'N'
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    i = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[i]

def temp_desc_c(t):
    if not np.isfinite(t):
        return 'Unknown'
    if t <= -10: return 'Frigid'
    if t <= 0: return 'Freezing'
    if t <= 10: return 'Cold'
    if t <= 18: return 'Cool'
    if t <= 24: return 'Mild'
    if t <= 30: return 'Warm'
    if t <= 35: return 'Hot'
    return 'Scorching'

def humidity_desc(p):
    if not np.isfinite(p):
        return 'Unknown'
    if p < 30: return 'Dry'
    if p < 50: return 'Comfortable'
    if p < 65: return 'Humid'
    return 'Oppressive'

frames = []
for frame_idx, (d, g) in enumerate(agg.groupby('frame_str')):

    # 城市桥梁曲线：所有城市两两连接，二次贝塞尔曲线
    def bridge_bezier(lon1, lat1, lon2, lat2, bulge=0.18, n=32):
        # 控制点为两点中点，向上（纬度）偏移 bulge*距离
        mx, my = (lon1+lon2)/2, (lat1+lat2)/2
        dx, dy = lon2-lon1, lat2-lat1
        # 计算正交方向（球面近似，向北偏 bulge*距离）
        norm = np.hypot(dx, dy)+1e-9
        nx, ny = -dy/norm, dx/norm
        # 控制点偏移量
        bulge_km = bulge * np.hypot((lon2-lon1)*R, (lat2-lat1)*R)
        ctrl_lon = mx + nx * bulge_km / (R * max(0.2, np.cos(np.deg2rad(my))))
        ctrl_lat = my + ny * bulge_km / R
        t = np.linspace(0,1,n)
        bx = (1-t)**2*lon1 + 2*(1-t)*t*ctrl_lon + t**2*lon2
        by = (1-t)**2*lat1 + 2*(1-t)*t*ctrl_lat + t**2*lat2
        return bx, by

    bridge_lon, bridge_lat = [], []
    city_lons = g['lon'].values
    city_lats = g['lat'].values
    n_city = len(city_lons)
    for i in range(n_city):
        for j in range(i+1, n_city):
            bx, by = bridge_bezier(city_lons[i], city_lats[i], city_lons[j], city_lats[j], bulge=0.18, n=32)
            bridge_lon += bx.tolist() + [None]
            bridge_lat += by.tolist() + [None]
    bridge_trace = go.Scattergeo(lon=bridge_lon, lat=bridge_lat, mode='lines',
                                 line=dict(color='rgba(255,255,255,0.22)', width=0.2),
                                 name='City Bridges', showlegend=False)
    # Prepare humidity arrays for consistent usage in tooltips
    hvals = pd.to_numeric(g['humidity'], errors='coerce')
    hvals = hvals.fillna(hvals.mean()).fillna(0.0)
    hmax = float(hvals.max()) if np.isfinite(hvals.max()) and hvals.max() > 0 else 1.0
    hsize = (hvals / hmax * 18.0 + 6.0)

    if MAPBOX_TOKEN:
        humidity_layer = go.Densitymapbox(
            lat=g['lat'], lon=g['lon'], z=g['humidity'],
            radius=40, colorscale='Blues', opacity=0.35,
            name='Humidity', showscale=True,
            colorbar=dict(title='Relative Humidity (%)', x=1.02)
        )
    else:
        humidity_layer = go.Scattergeo(
            lat=g['lat'], lon=g['lon'],
            marker=dict(
                size=hsize,
                color=hvals,
                colorscale='Blues', opacity=0.28,
                colorbar=dict(title='Relative Humidity (%)')
            ),
            mode='markers', name='Humidity (points)',
            showlegend=False
        )

    # Robust wind-based sizes (base +2 units bigger)
    wvals = pd.to_numeric(g['wind'], errors='coerce').fillna(0.0)
    wmax = float(wvals.max()) if np.isfinite(wvals.max()) and wvals.max() > 0 else 1.0
    wsize = (wvals / wmax * 14.0 + 8.0)

    # Weather -> symbol mapping
    def map_symbol(w):
        w = (str(w) or '').lower()
        if 'thunder' in w:
            return 'star'
        if 'snow' in w:
            return 'diamond'
        if 'drizzle' in w:
            return 'triangle-down'
        if 'rain' in w:
            return 'triangle-up'
        if 'fog' in w or 'mist' in w or 'haze' in w:
            return 'x'
        if 'cloud' in w:
            return 'square'
        if 'clear' in w:
            return 'circle'
        return 'circle'
    weather_raw = g.get('weather', pd.Series(['']*len(g))).tolist()
    symbols = [map_symbol(x) for x in weather_raw]
    # Build richer English annotations
    tvals = pd.to_numeric(g['temp'], errors='coerce').astype(float)
    tF = tvals * 9.0/5.0 + 32.0
    wdir = pd.to_numeric(g.get('wind_deg', pd.Series([np.nan]*len(g))), errors='coerce').astype(float)
    wdir_card = [deg_to_cardinal(x) for x in wdir]
    wkmh = wvals * 3.6
    wmph = wvals * 2.23693629
    tdesc = [temp_desc_c(x) for x in tvals]
    hdesc = [humidity_desc(x) for x in hvals]

    # compute representative color hex of each point based on colormap and temp
    # use our palette function for consistency
    color_hex = [_rgb_to_hex(temp_to_rgb(v, t_min, t_max)) for v in tvals]
    symbol_names = symbols

    temp_points = go.Scattergeo(
        lat=g['lat'], lon=g['lon'], text=g['city'],
        marker=dict(
            size=wsize,
            color=g['temp'], colorscale=[[0,'#003CFF'], [0.25,'#00C8FF'], [0.5,'#00DC78'], [0.75,'#FFE61E'], [1,'#FF2800']],
            cmin=t_min, cmax=t_max, line=dict(width=0), symbol=symbols,
            colorbar=dict(
                title='Temperature (°C)', orientation='h',
                x=0.5, xanchor='center', y=-0.14, yanchor='top',
                thickness=10, len=0.5
            )
        ),
        hovertemplate=(
            'City: %{text}'
            '<br>Temp: %{customdata[6]:.1f}°C (%{customdata[7]:.1f}°F) — %{customdata[9]}'
            '<br>Color: %{customdata[11]} (by temperature)'
            '<br>Humidity: %{customdata[0]:.0f}% — %{customdata[10]}'
            '<br>Wind: %{customdata[1]:.1f} m/s (%{customdata[2]:.1f} km/h, %{customdata[3]:.1f} mph) — %{customdata[5]} (%{customdata[4]:.0f}°)'
            '<br>Shape: %{customdata[12]} — Weather: %{customdata[8]}'
        ),
        customdata=np.c_[
            hvals.values,           # 0 humidity %
            wvals.values,           # 1 wind m/s
            wkmh.values,            # 2 wind km/h
            wmph.values,            # 3 wind mph
            wdir.values,            # 4 wind degrees
            np.array(wdir_card, dtype=object),  # 5 wind cardinal
            tvals.values,           # 6 temp C
            tF.values,              # 7 temp F
            np.array(g.get('weather', pd.Series(['']*len(g))).tolist(), dtype=object),  # 8 raw weather string
            np.array(tdesc, dtype=object),      # 9 temp desc
            np.array(hdesc, dtype=object),      # 10 humidity desc
            np.array(color_hex, dtype=object),  # 11 color hex by temperature
            np.array(symbol_names, dtype=object)  # 12 symbol/shape name
        ],
        mode='markers', name='Temperature'
    )

    # Wind-speed-driven glow intensity: split into configurable bins (continuous-like)
    wbin_vals = pd.to_numeric(g['wind'], errors='coerce').fillna(0.0).values
    wmin, wmax = float(np.nanmin(wbin_vals)), float(np.nanmax(wbin_vals))
    if not np.isfinite(wmin):
        wmin = 0.0
    if not np.isfinite(wmax) or wmax <= wmin:
        wmax = wmin + 1.0
    n_bins = max(1, int(args.glow_bins))
    edges = np.linspace(wmin, wmax, n_bins + 1)
    bin_traces = []

    def seg_arrays(idx):
        lon1 = g['lon'].values[idx]; lat1 = g['lat'].values[idx]
        lon2 = g['lon2'].values[idx]; lat2 = g['lat2'].values[idx]
        lon_pairs = np.c_[lon1, lon2].ravel(order='C').tolist()
        lat_pairs = np.c_[lat1, lat2].ravel(order='C').tolist()
        # insert None separators
        out_lon, out_lat = [], []
        for i in range(0, len(lon_pairs), 2):
            out_lon += [lon_pairs[i], lon_pairs[i+1], None]
            out_lat += [lat_pairs[i], lat_pairs[i+1], None]
        return out_lon, out_lat

    def head_arrays(idx):
        lon2 = g['lon2'].values[idx]; lat2 = g['lat2'].values[idx]
        lonh1 = g['lonh1'].values[idx]; lath1 = g['lath1'].values[idx]
        lonh2 = g['lonh2'].values[idx]; lath2 = g['lath2'].values[idx]
        out_lon, out_lat = [], []
        for i in range(len(idx)):
            out_lon += [lon2[i], lonh1[i], None, lon2[i], lonh2[i], None]
            out_lat += [lat2[i], lath1[i], None, lat2[i], lath2[i], None]
        return out_lon, out_lat

    for b in range(n_bins):
        mask = (wbin_vals >= edges[b]) & (wbin_vals <= edges[b+1]) if b == n_bins - 1 else ((wbin_vals >= edges[b]) & (wbin_vals < edges[b+1]))
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        # normalized intensity by bin center
        center = 0.5 * (edges[b] + edges[b+1])
        t = (center - wmin) / (wmax - wmin + 1e-6)
        # alpha mapping (smooth)
        a_outer = 0.10 + 0.35 * t
        a_mid = 0.16 + 0.44 * t
        # widths (smooth, slightly thinner at low wind), scaled globally
        aws = max(0.01, float(getattr(args, 'arrow_width_scale', 1.0)))
        w_outer = (4.5 + 3.0 * t) * aws
        w_mid = (2.5 + 1.5 * t) * aws
        w_main = (1.2 + 0.8 * t) * aws
        # colors
        col_outer = f'rgba(0,200,255,{a_outer:.3f})'
        col_mid = f'rgba(150,230,255,{a_mid:.3f})'
        col_main = 'rgba(255,255,255,0.98)'

        slon, slat = seg_arrays(idx)
        hlon, hlat = head_arrays(idx)

        # Glow layers and main for shafts
        bin_traces.append(go.Scattergeo(lon=slon, lat=slat, mode='lines', line=dict(color=col_outer, width=w_outer), showlegend=False))
        bin_traces.append(go.Scattergeo(lon=slon, lat=slat, mode='lines', line=dict(color=col_mid, width=w_mid), showlegend=False))
        bin_traces.append(go.Scattergeo(lon=slon, lat=slat, mode='lines', line=dict(color=col_main, width=w_main), name='Wind Direction' if b==n_bins//2 else None, showlegend=(b==n_bins//2)))
        # Heads
        bin_traces.append(go.Scattergeo(lon=hlon, lat=hlat, mode='lines', line=dict(color=col_outer, width=w_outer), showlegend=False))
        bin_traces.append(go.Scattergeo(lon=hlon, lat=hlat, mode='lines', line=dict(color=col_mid, width=w_mid), showlegend=False))
        bin_traces.append(go.Scattergeo(lon=hlon, lat=hlat, mode='lines', line=dict(color=col_main, width=w_main), showlegend=False))

    # Radiating gradient curves per city (combined into 3 traces per frame)
    rays_traces = []
    if not args.no_rays and args.rays > 0:
        n_color_bins = max(2, int(args.ray_color_bins))
        temp_edges = np.linspace(t_min, t_max, n_color_bins + 1)
        outer_lon_bins = [[] for _ in range(n_color_bins)]
        outer_lat_bins = [[] for _ in range(n_color_bins)]
        mid_lon_bins = [[] for _ in range(n_color_bins)]
        mid_lat_bins = [[] for _ in range(n_color_bins)]
        inner_lon_bins = [[] for _ in range(n_color_bins)]
        inner_lat_bins = [[] for _ in range(n_color_bins)]
        tail_lon, tail_lat, tail_color, tail_size_arr = [], [], [], []

        def add_segment(lst_lon, lst_lat, p_lon, p_lat):
            for i in range(len(p_lon)-1):
                lst_lon += [p_lon[i], p_lon[i+1], None]
                lst_lat += [p_lat[i], p_lat[i+1], None]

        def add_partial_segment(lst_lon, lst_lat, bx, by, frac):
            n = len(bx)
            if n < 2:
                return
            m = max(2, int(np.clip(frac, 0.0, 1.0) * n))
            # ensure at least a small visible portion
            m = max(m, 2)
            for i in range(m-1):
                lst_lon += [bx[i], bx[i+1], None]
                lst_lat += [by[i], by[i+1], None]

        def km_to_deg(lat_deg, dist_km, ang_rad):
            R = 111.0
            lat_rad = np.deg2rad(lat_deg)
            dlat = (dist_km / R) * np.cos(ang_rad)
            dlon = (dist_km / (R * max(0.2, np.cos(lat_rad)))) * np.sin(ang_rad)
            return dlon, dlat

        def quad_bezier(p0, p1, p2, n):
            t = np.linspace(0, 1, n)
            bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            return bx.tolist(), by.tolist()

        def apply_undulation(lon_list, lat_list, lat_ref, phase_shift):
            if not args.ray_undulate:
                return lon_list, lat_list
            bx = np.asarray(lon_list, dtype=float)
            by = np.asarray(lat_list, dtype=float)
            n = bx.size
            if n < 3:
                return lon_list, lat_list
            # compute normalized arc-length s in [0,1]
            # local km scaling
            Rloc = 111.0
            coslr = np.cos(np.deg2rad(lat_ref))
            dx = np.diff(bx) * (Rloc * max(0.2, coslr))
            dy = np.diff(by) * Rloc
            seg = np.sqrt(dx*dx + dy*dy)
            Ltot = float(np.sum(seg))
            if Ltot <= 1e-9:
                return lon_list, lat_list
            s = np.r_[0.0, np.cumsum(seg) / Ltot]
            # tangents and normals in lon/lat degrees
            tx = np.r_[bx[1]-bx[0], bx[1:]-bx[:-1]]
            ty = np.r_[by[1]-by[0], by[1:]-by[:-1]]
            # normalize to get direction
            scale = np.hypot(tx * (Rloc * max(0.2, coslr)), ty * Rloc) + 1e-9
            txn = tx / scale
            tyn = ty / scale
            # left normal (-ty, tx)
            nx = -tyn
            ny = txn
            # amplitude profile and sinusoidal term
            A = float(args.ray_undulate_amp_km)
            decay = float(args.ray_undulate_decay)
            waves = float(args.ray_undulate_waves)
            per = max(1e-6, float(args.ray_undulate_period))
            # time phase from frame index + per-ray shift
            tphase = 2*np.pi * (frame_idx / per) + phase_shift
            amp = A * (s ** decay)
            sinus = np.sin(2*np.pi*waves*s + tphase)
            off_km = amp * sinus
            # convert km normal offset back to degrees
            dlon = (off_km / (Rloc * max(0.2, coslr))) * nx
            dlat = (off_km / Rloc) * ny
            bx2 = bx + dlon
            by2 = by + dlat
            return bx2.tolist(), by2.tolist()

        # scale by wind within frame
        g_sorted = g.copy()
        # Optionally limit to top-K windy cities in this frame
        if args.rays_topk is not None and args.rays_topk > 0:
            g_sorted = g_sorted.sort_values('wind', ascending=False).head(args.rays_topk)
        # Wind stats (use full frame for stable normalization)
        wvals_frame_full = pd.to_numeric(g['wind'], errors='coerce').fillna(0.0)
        wmaxf = float(wvals_frame_full.max()) if np.isfinite(wvals_frame_full.max()) and wvals_frame_full.max() > 0 else 1.0

        # KNN-based target length per city (computed on full frame, then merged)
        if args.ray_length_mode == 'knn' and len(g) > 1:
            lat_arr = g['lat'].to_numpy(dtype=float)
            lon_arr = g['lon'].to_numpy(dtype=float)
            lat_rad = np.deg2rad(lat_arr)
            cos_lat = np.cos(lat_rad)
            N = len(g)
            dmat = np.zeros((N, N), dtype=float)
            for i in range(N):
                # approximate km distance using local scaling for longitude
                dx = (lon_arr - lon_arr[i]) * (R * np.maximum(0.2, cos_lat[i]))
                dy = (lat_arr - lat_arr[i]) * R
                dist_vec = np.sqrt(dx*dx + dy*dy)
                dist_vec[i] = np.inf
                dmat[i] = dist_vec
            k = max(1, min(int(args.ray_neighbor_k), N-1))
            d_sorted = np.sort(dmat, axis=1)
            dk = d_sorted[:, k-1]
            L_base = np.clip(dk * float(args.ray_neighbor_frac), float(args.ray_length_min_km), float(args.ray_length_max_km))
            g_knn = g.copy()
            g_knn['__ray_len_knn'] = L_base
            g_sorted = g_sorted.merge(g_knn[['city','lat','lon','__ray_len_knn']], on=['city','lat','lon'], how='left')
        else:
            g_sorted = g_sorted.copy()
            g_sorted['__ray_len_knn'] = np.nan

        rng = np.random.default_rng(42 + frame_idx)

        def add_tail_points_from_path(bx, by, tval, tip_idx=None):
            base_size = max(1.0, float(args.tail_size) * float(getattr(args, 'tail_size_scale', 1.0)) * 0.5)  # 缩小为原来一半
            n = len(bx)
            if n == 0:
                return
            if tip_idx is None:
                idxs = [n-1, max(n-2, 0), max(n-3, 0), max(n-4, 0)]
            else:
                ti = int(np.clip(tip_idx, 0, n-1))
                idxs = [ti, max(ti-1, 0), max(ti-2, 0), max(ti-3, 0)]
            sizes = [base_size, base_size/3.0, base_size/3.0, base_size/3.0]
            for ii, sz in zip(idxs, sizes):
                tail_lon.append(bx[ii]); tail_lat.append(by[ii])
                phi = 2*np.pi*rng.random()
                alpha = np.clip(0.3 + 0.7*np.sin(float(args.tail_omega)*frame_idx + phi), 0.15, 1.0) * float(args.tail_opacity)
                tail_color.append(_rgb_to_rgba_str(temp_to_rgb(tval, t_min, t_max), alpha))
                tail_size_arr.append(sz)
            # 随机增加无序闪烁小圆点
            n_extra = rng.integers(2, 5)  # 每帧2~4个
            n = len(bx)
            for _ in range(n_extra):
                idx = rng.integers(max(0, n-5), n)  # 尾部附近
                offset = rng.normal(0, 0.0008)  # 经纬度微扰
                tail_lon.append(bx[idx] + offset)
                tail_lat.append(by[idx] + offset)
                phi2 = 2*np.pi*rng.random()
                alpha2 = np.clip(0.2 + 0.8*np.sin(float(args.tail_omega)*frame_idx + phi2 + rng.random()*2), 0.08, 0.7) * float(args.tail_opacity)
                tail_color.append(_rgb_to_rgba_str(temp_to_rgb(tval, t_min, t_max), alpha2))
                tail_size_arr.append(base_size * rng.uniform(0.3, 0.7))
        for i, row in g_sorted.reset_index(drop=True).iterrows():
            lon0 = float(row['lon']); lat0 = float(row['lat'])
            wind = float(row['wind']) if pd.notna(row['wind']) else 0.0
            wscale = 0.6 + 0.4 * (wind / wmaxf)
            if args.ray_length_mode == 'knn' and pd.notna(row.get('__ray_len_knn', np.nan)):
                L0 = float(row['__ray_len_knn'])
                # enhance length with wind but keep at least L0
                L = L0 * (1.0 + 0.4 * (wind / max(wmaxf, 1e-6)))
                L = float(np.clip(L, args.ray_length_min_km, args.ray_length_max_km))
            else:
                L = args.ray_length_km * wscale
            # apply global length multiplier
            L *= max(0.01, float(args.ray_length_mult))
            # scale curvature with FINAL length to preserve visible curvature proportion
            C = args.ray_curvature_km * (L / max(1e-6, float(args.ray_length_km)))
            base = float(row['wind_deg']) if pd.notna(row['wind_deg']) else 0.0
            base_rad = np.deg2rad(base)
            tval = float(row['temp']) if 'temp' in row and pd.notna(row['temp']) else (t_min + t_max) * 0.5
            # temp bin index for per-bin coloring
            b_idx = int(np.clip(np.searchsorted(temp_edges, tval, side='right') - 1, 0, n_color_bins - 1))

            if args.rays_mode == 'three':
                angles = [base_rad, base_rad + np.deg2rad(args.ray_angle_deg), base_rad - np.deg2rad(args.ray_angle_deg)]
                for j, ang in enumerate(angles):
                    dlon_end, dlat_end = km_to_deg(lat0, L, ang)
                    lon2 = lon0 + dlon_end; lat2 = lat0 + dlat_end
                    ang_n = ang + np.pi/2.0
                    dlon_c, dlat_c = km_to_deg(lat0, C, ang_n)
                    lon1 = lon0 + (dlon_end*0.5) + dlon_c
                    lat1 = lat0 + (dlat_end*0.5) + dlat_c
                    bx, by = quad_bezier((lon0, lat0), (lon1, lat1), (lon2, lat2), max(4, args.ray_segments))
                    # apply jellyfish-like undulation along the ray
                    bx, by = apply_undulation(bx, by, lat0, phase_shift=0.6*j + rng.random()*0.5)
                    if args.ray_emit:
                        base_phase = rng.random() * float(args.ray_emit_phase_jitter)
                        speed = 0.6 + 1.4 * (wind / max(wmaxf, 1e-6))
                        period_eff = max(1.0, float(args.ray_emit_period)) / speed
                        phase = (frame_idx / period_eff + base_phase + 0.33*j) % 1.0
                        frac = max(args.ray_emit_min_frac, phase)
                        add_partial_segment(outer_lon_bins[b_idx], outer_lat_bins[b_idx], bx, by, frac)
                        add_partial_segment(mid_lon_bins[b_idx], mid_lat_bins[b_idx], bx, by, frac)
                        add_partial_segment(inner_lon_bins[b_idx], inner_lat_bins[b_idx], bx, by, frac)
                        # current tip position
                        tip_idx = max(0, min(len(bx)-1, int(frac * (len(bx)-1))))
                        add_tail_points_from_path(bx, by, tval, tip_idx=tip_idx)
                    else:
                        add_segment(outer_lon_bins[b_idx], outer_lat_bins[b_idx], bx, by)
                        add_segment(mid_lon_bins[b_idx], mid_lat_bins[b_idx], bx, by)
                        add_segment(inner_lon_bins[b_idx], inner_lat_bins[b_idx], bx, by)
                        add_tail_points_from_path(bx, by, tval, tip_idx=None)
            else:
                # ray decimation and optional rotation across frames
                step = max(1, int(args.rays_step))
                offset = (frame_idx % step) if args.rays_rotate else 0
                for k in range(offset, args.rays, step):
                    ang = base_rad + 2*np.pi * (k / args.rays)
                    dlon_end, dlat_end = km_to_deg(lat0, L, ang)
                    lon2 = lon0 + dlon_end; lat2 = lat0 + dlat_end
                    ang_n = ang + np.pi/2.0
                    dlon_c, dlat_c = km_to_deg(lat0, C, ang_n)
                    lon1 = lon0 + (dlon_end*0.5) + dlon_c
                    lat1 = lat0 + (dlat_end*0.5) + dlat_c
                    bx, by = quad_bezier((lon0, lat0), (lon1, lat1), (lon2, lat2), max(4, args.ray_segments))
                    bx, by = apply_undulation(bx, by, lat0, phase_shift=rng.random()*0.5)
                    if args.ray_emit:
                        base_phase = rng.random() * float(args.ray_emit_phase_jitter)
                        speed = 0.6 + 1.4 * (wind / max(wmaxf, 1e-6))
                        period_eff = max(1.0, float(args.ray_emit_period)) / speed
                        phase = (frame_idx / period_eff + base_phase) % 1.0
                        frac = max(args.ray_emit_min_frac, phase)
                        add_partial_segment(outer_lon_bins[b_idx], outer_lat_bins[b_idx], bx, by, frac)
                        add_partial_segment(mid_lon_bins[b_idx], mid_lat_bins[b_idx], bx, by, frac)
                        add_partial_segment(inner_lon_bins[b_idx], inner_lat_bins[b_idx], bx, by, frac)
                        tip_idx = max(0, min(len(bx)-1, int(frac * (len(bx)-1))))
                        add_tail_points_from_path(bx, by, tval, tip_idx=tip_idx)
                    else:
                        add_segment(outer_lon_bins[b_idx], outer_lat_bins[b_idx], bx, by)
                        add_segment(mid_lon_bins[b_idx], mid_lat_bins[b_idx], bx, by)
                        add_segment(inner_lon_bins[b_idx], inner_lat_bins[b_idx], bx, by)
                        add_tail_points_from_path(bx, by, tval, tip_idx=None)

        # Build three layered traces for gradient look
        outer_col = 'rgba(0,120,255,0.18)'
        mid_col = 'rgba(120,220,255,0.35)'
        inner_col = 'rgba(255,255,255,0.95)'
        ws = max(0.05, float(args.ray_width_scale))
        # Build per-temperature-bin traces with gradient glow derived from bin color
        for bi in range(n_color_bins):
            # representative color of bin center
            tc = 0.5*(temp_edges[bi] + temp_edges[bi+1])
            base_rgb = temp_to_rgb(tc, t_min, t_max)
            mid_rgb = lighten_rgb(base_rgb, 0.45)
            outer_rgb = lighten_rgb(base_rgb, 0.75)
            inner_col = _rgb_to_rgba_str(base_rgb, 0.96)
            mid_col = _rgb_to_rgba_str(mid_rgb, 0.38)
            outer_col = _rgb_to_rgba_str(outer_rgb, 0.18)
            rays_traces.append(go.Scattergeo(lon=outer_lon_bins[bi], lat=outer_lat_bins[bi], mode='lines', line=dict(color=outer_col, width=ws*max(1.0, args.ray_base_width+4)), showlegend=False))
            rays_traces.append(go.Scattergeo(lon=mid_lon_bins[bi], lat=mid_lat_bins[bi], mode='lines', line=dict(color=mid_col, width=ws*max(1.0, args.ray_base_width+2)), showlegend=False))
            rays_traces.append(go.Scattergeo(lon=inner_lon_bins[bi], lat=inner_lat_bins[bi], mode='lines', line=dict(color=inner_col, width=ws*max(0.5, args.ray_base_width)), showlegend=False))
        if args.tail_blink and len(tail_lon) > 0:
            # blinking with per-tail alpha/colors already computed
            tail_marker = dict(size=tail_size_arr, color=tail_color, line=dict(width=0))
            rays_traces.append(go.Scattergeo(lon=tail_lon, lat=tail_lat, mode='markers', marker=tail_marker, name='Ray Tails', showlegend=False))

    # Optional firework particles overlay
    firework_traces = []
    if getattr(args, 'firework', False):
        rng_fw = np.random.default_rng(2025 + frame_idx)
        sparks_lon, sparks_lat, sparks_col, sparks_size = [], [], [], []
        trails_lon, trails_lat, trails_col, trails_size = [], [], [], []
        core_lon, core_lat, core_col, core_size = [], [], [], []
        # Use all cities or topK windy cities for bursts (reuse rays_topk logic)
        g_fw = g.copy()
        if args.rays_topk is not None and args.rays_topk > 0:
            g_fw = g_fw.sort_values('wind', ascending=False).head(args.rays_topk)
        for _, rw in g_fw.iterrows():
            latc = float(rw['lat']); lonc = float(rw['lon'])
            tval = float(rw['temp']) if pd.notna(rw['temp']) else (t_min + t_max) * 0.5
            base_rgb = temp_to_rgb(tval, t_min, t_max)
            # per-city phase
            base_phase = rng_fw.random() * float(args.fw_burst_jitter)
            p = (frame_idx / max(1, int(args.fw_burst_period)) + base_phase) % 1.0
            # smoother start/end using ease in/out
            ease = 0.5 - 0.5 * np.cos(np.pi * p)
            Rkm = float(args.fw_expansion_km) * ease
            # size and alpha by progress
            alpha = float(np.clip((1.0 - p) ** float(args.fw_fade_power), 0.05, 0.95))
            inner_rgb = lighten_rgb(base_rgb, 0.15)
            col_rgba = _rgb_to_rgba_str(inner_rgb, alpha)
            nsp = max(3, int(args.fw_sparks))
            # random slight jitter per spark
            ang = np.linspace(0, 2*np.pi, nsp, endpoint=False) + rng_fw.uniform(-0.06, 0.06, size=nsp)
            # convert km to degrees around city
            coslr = np.cos(np.deg2rad(latc))
            dlon = (Rkm / (R * max(0.2, coslr))) * np.sin(ang)
            dlat = (Rkm / R) * np.cos(ang)
            lx = (lonc + dlon)
            ly = (latc + dlat)
            sparks_lon += lx.tolist()
            sparks_lat += ly.tolist()
            sparks_col += [col_rgba] * nsp
            sparks_size += [float(args.fw_spark_size)] * nsp

            # Trails behind current progress
            if getattr(args, 'fw_trail', False) and int(args.fw_trail_samples) > 0 and args.fw_trail_span > 0:
                ns = int(args.fw_trail_samples)
                span = float(args.fw_trail_span)
                for si in range(1, ns+1):
                    pf = np.clip(p - span * (si/ns), 0.0, 1.0)
                    ease_f = 0.5 - 0.5 * np.cos(np.pi * pf)
                    Rkm_f = float(args.fw_expansion_km) * ease_f
                    alpha_f = float(np.clip((1.0 - pf) ** (float(args.fw_fade_power) + float(args.fw_trail_fade)), 0.02, 0.7))
                    size_f = float(args.fw_spark_size) * float(args.fw_trail_size_scale)
                    dlon_f = (Rkm_f / (R * max(0.2, coslr))) * np.sin(ang)
                    dlat_f = (Rkm_f / R) * np.cos(ang)
                    trails_lon += (lonc + dlon_f).tolist()
                    trails_lat += (latc + dlat_f).tolist()
                    trails_col += [_rgb_to_rgba_str(inner_rgb, alpha_f)] * nsp
                    trails_size += [size_f] * nsp

            # Core flash at the center
            if getattr(args, 'fw_core', False) and p <= float(args.fw_core_duration):
                core_lon.append(lonc)
                core_lat.append(latc)
                core_col.append(_rgb_to_rgba_str(lighten_rgb(base_rgb, 0.6), float(args.fw_core_alpha)))
                core_size.append(float(args.fw_core_size))
        if sparks_lon:
            firework_traces.append(go.Scattergeo(lon=sparks_lon, lat=sparks_lat, mode='markers',
                                                 marker=dict(size=sparks_size, color=sparks_col, line=dict(width=0)),
                                                 name='Firework', showlegend=False))
        if trails_lon:
            firework_traces.append(go.Scattergeo(lon=trails_lon, lat=trails_lat, mode='markers',
                                                 marker=dict(size=trails_size, color=trails_col, line=dict(width=0)),
                                                 name='Firework Trails', showlegend=False))
        if core_lon:
            firework_traces.append(go.Scattergeo(lon=core_lon, lat=core_lat, mode='markers',
                                                 marker=dict(size=core_size, color=core_col, line=dict(width=0)),
                                                 name='Firework Core', showlegend=False))

    # Legend-only shape mapping traces so bottom legend shows shape=>weather
    legend_shapes = []
    shape_map = [
        ('Clear', 'circle'),
        ('Clouds', 'square'),
        ('Rain', 'triangle-up'),
        ('Drizzle', 'triangle-down'),
        ('Snow', 'diamond'),
        ('Thunder', 'star'),
        ('Fog/Mist/Haze', 'x')
    ]
    for lbl, sym in shape_map:
        legend_shapes.append(go.Scattergeo(
            lon=[0], lat=[0], mode='markers',
            marker=dict(size=12, symbol=sym, color='rgba(220,220,220,0.95)'),
            name=lbl, showlegend=True, visible='legendonly', hoverinfo='skip'
        ))

    frames.append(go.Frame(data=[humidity_layer, temp_points, bridge_trace] + bin_traces + rays_traces + firework_traces + legend_shapes, name=d))

first = frames[0].data if frames else []
fig = go.Figure(data=first, frames=frames)
fig.update_layout(
    title=f"Global Cities Weather — Last 30 Days ({'Daily Avg' if args.granularity=='daily' else 'Hourly'})\nHumidity base + Wind arrows",
    geo=dict(
        projection_type='natural earth', showcountries=True, showcoastlines=True, showland=True,
        landcolor='#111', bgcolor='#000', lakecolor='#000'
    ),
    paper_bgcolor='#000', plot_bgcolor='#000', font=dict(color='#EEE'),
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.08, yanchor='top',
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=0, r=0, t=80, b=90),
    updatemenus=[
        {
            'type': 'buttons', 'direction': 'left', 'x': 0.02, 'y': 0.02,
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': int(1000/max(args.fps,1)), 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ]
        },
        {
            'type': 'buttons', 'direction': 'left', 'x': 0.02, 'y': 0.08,
            'buttons': [
                {'label': 'Toggle Humidity', 'method': 'relayout', 'args': [{'visible': [True, None, None, None]}]},
            ], 'showactive': False
        }
    ],
    annotations=[
        dict(
            x=0.5, y=-0.18, xref='paper', yref='paper', xanchor='center', yanchor='top',
            text=('Color encodes Temperature (°C) from cold (blue) to hot (red). '
                  'Shape encodes Weather: circle=Clear, square=Clouds, triangle-up=Rain, '
                  'triangle-down=Drizzle, diamond=Snow, star=Thunder, x=Fog/Mist/Haze.'),
            showarrow=False, font=dict(color='#BBB', size=11)
        )
    ]
)

MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN', '')
if MAPBOX_TOKEN:
    fig.update_layout(mapbox_accesstoken=MAPBOX_TOKEN)

fig.write_html(str(OUTPUT_HTML), include_plotlyjs='cdn', auto_play=False)
print(f'Wrote {OUTPUT_HTML}')
 
