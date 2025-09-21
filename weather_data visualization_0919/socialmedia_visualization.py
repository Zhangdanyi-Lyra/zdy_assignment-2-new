import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# 六个数据文件路径
csv_files = [
    'data/statistics--Worldwide-2012---2024-Daily-time-spent-on-social-networking-by-internet-users-worldwide-from-2012-to-2024-in-minutes.csv',
    'data/statistics--Worldwide-2017---2028-Number-of-social-media-users-worldwide-from-2017-to-2028-in-billions---forecast.csv',
    'data/statistics--Worldwide-2017---2029-Social-Media-Advertising-global-ad-spending-and-ad-spending-change-from-2017-to-2029.csv',
    'data/statistics--Worldwide-2024-Instagram-accounts-with-the-most-followers-worldwide-as-of-April-2024-in-millions.csv',
    'data/statistics--Worldwide-2025-Most-followed-creators-on-TikTok-worldwide-as-of-January-2025-in-millions.csv',
    'data/statistics--worldwide-2025-Most-popular-social-networks-worldwide-as-of-February-2025-by-number-of-monthly-active-users-in-millions.csv'
]

# 赛博朋克风格高饱和霓虹色
colors = ['#00fff7', '#ff00ea', '#39ff14', '#fffb00', '#ff005b', '#00b3ff']

# 每种颜色对应的数据类型英文说明
data_labels = [
    'Daily Social Networking Time (min)',
    'Social Media Users (billion)',
    'Social Media Ad Spending (billion USD)',
    'Instagram Top Followers (million)',
    'TikTok Top Creators (million)',
    'Popular Social Networks (million MAU)'
]

# 对应logo图片路径（如无图片可用占位图）
logo_paths = [
    None,  # 1. 日均时长无logo
    None,  # 2. 用户数无logo
    None,  # 3. 广告支出无logo
    'logo/instagram.png',
    'logo/tiktok.png',
    'logo/facebook.png'  # 以Facebook为代表
]

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # 取第二行（Total）所有数值列
    row = df.iloc[0, 1:].values
    # 处理千分位逗号
    row = [float(str(x).replace(",", "")) for x in row]
    return np.array(row)

def normalize(arr, min_dist=0.5, max_dist=2.5):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.ones_like(arr) * ((min_dist + max_dist) / 2)
    norm = (arr.max() - arr) / (arr.max() - arr.min())
    return min_dist + norm * (max_dist - min_dist)

def get_saturations(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.ones_like(arr) * 0.7
    return 0.3 + 0.7 * (arr - arr.min()) / (arr.max() - arr.min())


def draw_single_dandelion(ax, center, datas, colors, labels, logos, frame, legend_handles=None):
    # datas: list of np.array, colors: list, labels: list, logos: list
    n_group = len(datas)
    theta_offset = 0
    for i, (data, color, label, logo) in enumerate(zip(datas, colors, labels, logos)):
        n = len(data)
        show_n = min(frame+1, n)
        d = data[:show_n]
        radii = normalize(d)
        saturations = get_saturations(d)
        angles = np.linspace(theta_offset, theta_offset + 2*np.pi/n*show_n, show_n, endpoint=False)
        for j, (r, a, s, val) in enumerate(zip(radii, angles, saturations, d)):
            # 主干自然曲线（贝塞尔扰动）
            num_main = 20
            # 随机扰动参数，保证每帧一致
            np.random.seed(i*1000+j)
            ctrl_frac = np.random.uniform(0.3, 0.7)
            ctrl_angle = a + np.random.uniform(-np.pi/6, np.pi/6)
            ctrl_r = r * ctrl_frac * np.random.uniform(0.7, 1.3)
            ctrl_x = center[0] + ctrl_r * np.cos(ctrl_angle)
            ctrl_y = center[1] + ctrl_r * np.sin(ctrl_angle)
            x_end = center[0] + r * np.cos(a)
            y_end = center[1] + r * np.sin(a)
            for frac in np.linspace(0, 1, num_main):
                # 二阶贝塞尔插值
                px = (1-frac)**2*center[0] + 2*(1-frac)*frac*ctrl_x + frac**2*x_end
                py = (1-frac)**2*center[1] + 2*(1-frac)*frac*ctrl_y + frac**2*y_end
                ax.scatter([px], [py], color=color, alpha=s*frac*0.8+0.2, s=10, edgecolors='none')
            # 花瓣末端主点
            x = x_end
            y = y_end
            ax.scatter([x], [y], color=color, alpha=s, s=60, edgecolors='k', linewidths=0.5, zorder=5)
            # 花瓣末端logo（仅社交媒体相关）
            if logo is not None:
                try:
                    img = mpimg.imread(logo)
                    imgbox = ax.inset_axes([0,0,0.08,0.08], transform=ax.transData)
                    imgbox.set_anchor('C')
                    imgbox.set_aspect('auto')
                    imgbox.set_xlim(-0.5,0.5)
                    imgbox.set_ylim(-0.5,0.5)
                    imgbox.axis('off')
                    imgbox.imshow(img, extent=[x-0.18, x+0.18, y-0.18, y+0.18], zorder=10)
                except Exception:
                    pass
            # 花瓣末端数值标注
            ax.text(x, y+0.18, f"{val:.2f}", ha='center', va='bottom', fontsize=8, color=color, alpha=0.85, zorder=10)
            # 末端丝线弥散
            num_silks = 8
            for silk in range(num_silks):
                silk_angle = a + np.random.uniform(-np.pi/18, np.pi/18)
                silk_len = np.random.uniform(0.3, 0.7)
                silk_r = r + silk_len
                silk_points = 8
                for frac in np.linspace(0, 1, silk_points):
                    sx = x + frac * (silk_r * np.cos(silk_angle) - x)
                    sy = y + frac * (silk_r * np.sin(silk_angle) - y)
                    ax.scatter([sx], [sy], color=color, alpha=s*0.2*(1-frac)+0.1, s=6, edgecolors='none', zorder=4)
        # 图例句柄
        if legend_handles is not None:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=label))
        theta_offset += 2*np.pi/n
    # 蒲公英中心
    ax.scatter([center[0]], [center[1]], color='gray', s=200, zorder=10)

def dandelion_frame(ax, all_data, frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    legend_handles = []
    # 取消背景点阵，赛博朋克黑色背景
    ax.set_facecolor('#181828')
    # 四朵蒲公英的中心
    centers = [(0,0), (3,2), (-3,2), (0,-3)]
    # 第一朵：前三个数据，三色花瓣
    draw_single_dandelion(
        ax,
        centers[0],
        all_data[:3],
        colors[:3],
        data_labels[:3],
        [None,None,None],
        frame,
        legend_handles
    )
    # 其余三朵：各自一个数据
    for i in range(3,6):
        draw_single_dandelion(
            ax,
            centers[i-2],
            [all_data[i]],
            [colors[i]],
            [data_labels[i]],
            [logo_paths[i]],
            frame,
            legend_handles
        )
    # 添加图例
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=9, frameon=False, title="Data Type")
    # 添加远近与颜色饱和度说明
    ax.text(0, -4.2, "Closer to center: Higher value\nFarther: Lower value\nColor saturation: Value magnitude", ha='center', va='top', fontsize=10, color='dimgray')

def main():
    all_data = [load_data(f) for f in csv_files]
    max_len = max(len(d) for d in all_data)
    fig, ax = plt.subplots(figsize=(7,7))
    def update(frame):
        dandelion_frame(ax, all_data, frame)
        ax.set_title(f"Data Analysis of Social Media Influencer Economy", fontsize=16)
    ani = FuncAnimation(fig, update, frames=max_len, interval=400, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()