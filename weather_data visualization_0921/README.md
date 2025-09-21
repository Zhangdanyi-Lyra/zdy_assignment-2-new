
# Weather Data Visualization 0921

This folder contains a comprehensive animated weather visualization system that creates stunning interactive maps displaying global weather patterns with advanced visual effects.

## Contents

- `world_weather_timeseries_map_wind.py`: Main Python script for generating the animated visualization
- `city_weather.csv`: Weather data for global cities (last 30 days) containing temperature, humidity, wind speed/direction
- `outputs/weather_rays_len6_firework_tailfx.html`: Final animation output (94MB HTML file with rich effects)
- `data/`: Additional data processing scripts

## Prerequisites

### Required Python Packages
```bash
pip install pandas numpy plotly
```

### System Requirements
- Python 3.7+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- At least 4GB RAM for smooth playback
- Graphics acceleration recommended for optimal performance

## Quick Start

### Basic Usage
Run the following command in this directory to generate the animated visualization:

```bash
python world_weather_timeseries_map_wind.py \
    --input city_weather.csv \
    --out outputs/weather_rays_len6_firework_tailfx.html \
    --fps 12 --rays 10 --ray-undulate --ray-emit \
    --arrow-scale-km 40 --arrow-width-scale 0.4 \
    --firework --fw-trail --fw-core \
    --ray-length-mult 6.0 --ray-segments 8 --fw-expansion-km 440.0
```

### Viewing the Animation
1. **Local HTTP Server (Recommended)**:
   ```bash
   cd outputs
   python3 -m http.server 8000
   ```
   Then open `http://localhost:8000/weather_rays_len6_firework_tailfx.html` in your browser.

2. **Direct File Opening**: 
   Double-click the HTML file or open it directly in your browser.

3. **VS Code Live Server**: 
   Install the Live Server extension and right-click the HTML file → "Open with Live Server".

## Visualization Features

### Core Weather Data
- **Temperature Mapping**: Color-coded from blue (cold) to red (hot)
- **Wind Vectors**: Arrows showing wind direction and speed
- **Humidity Levels**: Affects marker sizing and glow effects
- **Weather Conditions**: Different symbols for Clear, Clouds, Rain, Snow, etc.

### Advanced Visual Effects
- **Radiating Rays** (`--ray-undulate --ray-emit`): Dynamic curves emanating from cities
- **Firework Bursts** (`--firework --fw-trail --fw-core`): Explosive particle effects
- **Undulating Motion**: Jellyfish-like wave animations along rays
- **Emission Animation**: Rays growing outward from city centers
- **Tail Effects**: Blinking circles and trailing particles

### Interactive Controls
- **Play/Pause**: Control animation playback
- **Frame Navigation**: Step through time periods
- **Hover Information**: Detailed weather data tooltips
- **Toggle Features**: Show/hide different visualization layers

## Configuration Options

### Animation Parameters
- `--fps 12`: Frames per second (1-30)
- `--granularity daily|hourly`: Time resolution
- `--hour-step N`: Skip hours in hourly mode

### Visual Effects
- `--rays N`: Number of rays per city (default: 16)
- `--ray-undulate`: Enable wave motion
- `--ray-emit`: Enable emission animation
- `--firework`: Enable firework effects
- `--fw-trail`: Add particle trails
- `--fw-core`: Add bright core flash

### Styling
- `--arrow-scale-km N`: Wind arrow length scale
- `--ray-length-mult N`: Ray length multiplier
- `--fw-expansion-km N`: Firework explosion radius

### Performance Options
- `--lite`: Enable lightweight preset for smoother playback
- `--glow-bins N`: Glow resolution (higher = smoother)
- `--ray-segments N`: Ray smoothness (recommended: 8)

## Example Configurations

### High Performance (Smooth Playback)
```bash
python world_weather_timeseries_map_wind.py \
    --input city_weather.csv \
    --out outputs/weather_lite.html \
    --lite --fps 15
```

### Maximum Visual Effects
```bash
python world_weather_timeseries_map_wind.py \
    --input city_weather.csv \
    --out outputs/weather_full_fx.html \
    --fps 8 --rays 20 --ray-undulate --ray-emit \
    --firework --fw-trail --fw-core \
    --ray-length-mult 8.0 --fw-expansion-km 600.0 \
    --glow-bins 16 --ray-segments 12
```

### Minimal (Fast Generation)
```bash
python world_weather_timeseries_map_wind.py \
    --input city_weather.csv \
    --out outputs/weather_minimal.html \
    --no-rays --fps 20
```

## Troubleshooting

### Common Issues
1. **Large File Size**: The output HTML can be 50-100MB with full effects
2. **Slow Browser**: Reduce `--fps`, `--rays`, or use `--lite` mode
3. **Memory Usage**: Close other browser tabs for better performance
4. **Missing Data**: Ensure `city_weather.csv` contains required columns

### Performance Tips
- Use Chrome or Firefox for best performance
- Enable hardware acceleration in browser settings
- Close unnecessary applications while viewing
- Consider using `--lite` mode for older computers

## Data Format

The input CSV should contain these columns:
- `city`: City name
- `lat`, `lon`: Geographic coordinates
- `time`: Timestamp (ISO format)
- `temp`: Temperature in Celsius
- `humidity`: Humidity percentage
- `wind`: Wind speed in m/s
- `wind_deg`: Wind direction in degrees
- `weather`: Weather condition string

## Output Information

The generated HTML file is completely self-contained and includes:
- All visualization data embedded
- Plotly.js library via CDN
- Interactive controls
- Responsive design for different screen sizes
- Detailed tooltips with weather information in English

Enjoy exploring global weather patterns with this advanced visualization system!
