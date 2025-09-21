
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba




# 多朵花参数
flowers = [
    dict(petals=7, scale=1.0, angle_offset=0, color_start='#ffb6c1', color_end='#87cefa', center=(0, 0)), # 粉-蓝
    dict(petals=5, scale=0.7, angle_offset=np.pi/5, color_start='#ffa07a', color_end='#20b2aa', center=(0.7, 0.6)), # 橙-青
    dict(petals=9, scale=0.5, angle_offset=np.pi/2, color_start='#dda0dd', color_end='#4169e1', center=(-0.8, 0.5)), # 紫-蓝
    dict(petals=6, scale=0.85, angle_offset=-np.pi/4, color_start='#ffe4e1', color_end='#4682b4', center=(0.6, -0.7)), # 淡粉-钢蓝
    dict(petals=8, scale=0.6, angle_offset=np.pi/3, color_start='#98fb98', color_end='#ff69b4', center=(-0.7, -0.8)), # 绿-粉
]
np.random.seed(42)
flower_noise = [
    dict(
        freqs=np.random.uniform(0.8, 2.5, size=5),
        amps=np.random.uniform(0.08, 0.22, size=5),
        phases=np.random.uniform(0, 2*np.pi, size=5)
    ) for _ in flowers
]

# 每朵花的轨迹
all_xdata = [[] for _ in flowers]
all_ydata = [[] for _ in flowers]

fig, ax = plt.subplots()
lines = []
dots = []
for flower in flowers:
    line, = plt.plot([], [], linewidth=2.5, alpha=0.85)
    dot, = plt.plot([], [], 'o', color=flower['color_end'], markersize=8, alpha=0.9)
    lines.append(line)
    dots.append(dot)
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-1.7, 1.7)
ax.set_aspect('equal')
ax.set_title("Schiele风格多花艺术")

def init():
    for line, dot in zip(lines, dots):
        line.set_data([], [])
        dot.set_data([], [])
    return lines + dots

def update(frame):
    t = frame / 30
    loop = frame // 360
    step = 0.07
    artists = []
    for i, (flower, noise) in enumerate(zip(flowers, flower_noise)):
        petals = flower['petals']
        scale = flower['scale']
        angle_offset = flower['angle_offset']
        color_start = np.array(to_rgba(flower['color_start']))
        color_end = np.array(to_rgba(flower['color_end']))
        # 花瓣半径+噪声
        r = scale * (0.7 * np.sin(petals * t))
        for f, a, p in zip(noise['freqs'], noise['amps'], noise['phases']):
            r += a * np.sin(f * t + p)
        r += loop * step
        # 角度扰动
        angle_jitter = 0.18 * np.sin(2.3 * t + 1.2 + i) + 0.12 * np.cos(1.7 * t + 2.1 - i)
        center = flower.get('center', (0, 0))
        x = r * np.cos(t + angle_jitter + angle_offset) + center[0]
        y = r * np.sin(t + angle_jitter + angle_offset) + center[1]
        all_xdata[i].append(x)
        all_ydata[i].append(y)
        if len(all_xdata[i]) > 360:
            all_xdata[i].pop(0)
            all_ydata[i].pop(0)
        lines[i].set_data(all_xdata[i], all_ydata[i])
        dots[i].set_data([x], [y])
        # 渐变色
        if len(all_xdata[i]) > 1:
            ratio = (frame % 360) / 360
            color = color_start * (1 - ratio) + color_end * ratio
            lines[i].set_color(color)
        lines[i].set_linewidth(2.5 + 1.2 * np.sin(0.5 * t + i))
        artists.extend([lines[i], dots[i]])
    return artists

ani = FuncAnimation(fig, update, frames=range(0, 360*5), init_func=init, blit=True, interval=30, repeat=True)
plt.show()
