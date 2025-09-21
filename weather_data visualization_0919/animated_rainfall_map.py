import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap

# 示例数据生成（实际使用时请替换为真实数据读取）
def generate_sample_data():
    np.random.seed(0)
    years = np.arange(1925, 2025)
    months = np.arange(1, 13)
    lats = np.random.uniform(-60, 80, 200)
    lons = np.random.uniform(-180, 180, 200)
    data = []
    for year in years:
        for month in months:
            for lat, lon in zip(lats, lons):
                rainfall = np.random.gamma(2, 20)
                data.append([year, month, lat, lon, rainfall])
    df = pd.DataFrame(data, columns=["year", "month", "lat", "lon", "rainfall"])
    return df

def main():
    # df = pd.read_csv('rainfall_data.csv')  # 实际数据读取
    df = generate_sample_data()  # 示例数据
    years = sorted(df['year'].unique())
    months = sorted(df['month'].unique())
    frames = [(y, m) for y in years for m in months]

    fig, ax = plt.subplots(figsize=(10, 6))
    m = Basemap(projection='cyl', resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    scat = ax.scatter([], [], c=[], cmap='Blues', vmin=0, vmax=df['rainfall'].max(), s=20)
    title = ax.set_title("")

    def update(frame):
        year, month = frame
        subset = df[(df['year'] == year) & (df['month'] == month)]
        x, y = subset['lon'].values, subset['lat'].values
        c = subset['rainfall'].values
        scat.set_offsets(np.c_[x, y])
        scat.set_array(c)
        title.set_text(f"{year}年{month}月 平均降雨量")
        return scat, title

    ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False, repeat=True)
    plt.colorbar(scat, ax=ax, label='降雨量 (mm)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
