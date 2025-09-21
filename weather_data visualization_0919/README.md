Global Weather Visualization (Plotly)

- 功能: 基于 Plotly 的全球城市天气动画，可显示湿度底图、温度点、风向箭头、辐射曲线（支持水母尾巴起伏/发射效果）、烟花特效（火花/拖尾/核心闪光）与底部图例注释。现已支持导出 HTML 后自动注入背景音乐（BGM）。

快速开始
- 依赖: Python 3.x、pandas、numpy、plotly
- 数据: 需要 `city_weather.csv`（包含列: `time, city, lat, lon, temp, humidity, wind, wind_deg, weather`），时间范围建议近 30 天。
- 运行示例（写出 HTML，不自动播放动画）：
	`python world_weather_timeseries_map_wind.py --input city_weather.csv --out outputs/weather.html --fps 6`

背景音乐（BGM）
- 参数：
	- `--bgm <path_or_url>`: 音频文件路径或 URL（mp3/ogg）。
	- `--bgm-volume <0-1>`: 初始音量（默认 0.35）。
	- `--bgm-loop`: 循环播放。
	- `--bgm-autoplay`: 尝试自动播放（因浏览器策略将以静音启动）。
	- `--bgm-controls`: 显示音频控件（页面左下角，默认开启）。
	- `--bgm-unmute-onclick`: 首次点击页面时解除静音并开始播放。
- 示例：
	`python world_weather_timeseries_map_wind.py --input city_weather.csv --out outputs/weather_bgm.html --fps 6 --bgm data/bgm.mp3 --bgm-loop --bgm-autoplay --bgm-unmute-onclick --bgm-volume 0.4`
- 说明：导出的 HTML 底部会插入 `<audio>`，若使用 `--bgm-autoplay`，页面将以静音启动；配合 `--bgm-unmute-onclick` 可在首次点击时自动取消静音并播放。

其它常用参数
- 温度范围: `--tmin --tmax`
- 风箭头: `--arrow-scale-km --arrow-head-deg --arrow-head-scale --arrow-width-scale`
- 曲线辐射: `--rays --rays-mode [radial|three] --ray-length-mode [wind|knn] --ray-length-km --ray-length-mult --ray-curvature-km`
- 水母尾巴: `--ray-undulate --ray-undulate-amp-km --ray-undulate-waves --ray-undulate-period --ray-undulate-decay`
- 发射效果: `--ray-emit --ray-emit-period --ray-emit-phase-jitter --ray-emit-min-frac`
- 烟花特效: `--firework --fw-sparks --fw-burst-period --fw-burst-jitter --fw-expansion-km --fw-fade-power --fw-spark-size --fw-trail --fw-core`

提示
- 自动播放策略：大多数浏览器禁止未静音的自动播放，故 `--bgm-autoplay` 会以 muted 启动；建议与 `--bgm-unmute-onclick` 搭配。
- 输出文件：默认写至 `outputs/*.html`，可直接在浏览器打开预览。
