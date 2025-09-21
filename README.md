# Weather Data Visualization Assignment

**Course**: SD5913 - Advanced Data Science and Visualization  
**Author**: Zhangdanyi-Lyra  
**Date**: September 2025

## Project Overview

This repository contains advanced weather data visualization projects that demonstrate sophisticated data science techniques and interactive visualization capabilities. The project focuses on creating dynamic, animated weather maps with real-time data from global cities, featuring cutting-edge visual effects and user interaction.

## 🌟 Key Features

- **Real-time Weather Data**: 30-day historical weather data from major global cities
- **Advanced Animations**: Dynamic rays, firework effects, and fluid motion graphics
- **Interactive Controls**: Play/pause, frame navigation, and layer toggles
- **Multiple Visual Effects**: Temperature gradients, wind vectors, humidity mapping
- **Responsive Design**: Works across different devices and screen sizes
- **Performance Optimization**: Multiple rendering modes for different hardware capabilities

## 📁 Project Structure

```
zdy_assignment-2-new/
├── weather_data visualization_0919/     # Initial development version
│   ├── world_weather_timeseries_map_wind.py
│   ├── city_weather.csv
│   ├── animated_rainfall_map.py
│   ├── complex_animation.py
│   ├── socialmedia_visualization.py
│   ├── sprott_linz_attractor.py
│   ├── weather_network_animation.py
│   └── README.md
│
├── weather_data visualization_0921/     # Enhanced production version
│   ├── world_weather_timeseries_map_wind.py    # Main visualization script
│   ├── city_weather.csv                        # Global weather dataset
│   ├── data/                                   # Data processing utilities
│   │   ├── weather_data_fetch.py
│   │   ├── weather_forecast_viz.py
│   │   └── weather_timeseries_fetch.py
│   ├── outputs/
│   │   └── weather_rays_len6_firework_tailfx.html  # Generated visualization (94MB)
│   └── README.md                               # Detailed usage instructions
│
└── README.md                                   # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy plotly

# System requirements
- Python 3.7+
- Modern web browser
- 4GB+ RAM recommended
```

### Running the Visualization

1. **Navigate to the project directory:**
   ```bash
   cd "weather_data visualization_0921"
   ```

2. **Generate the visualization:**
   ```bash
   python world_weather_timeseries_map_wind.py \
       --input city_weather.csv \
       --out outputs/weather_rays_len6_firework_tailfx.html \
       --fps 12 --rays 10 --ray-undulate --ray-emit \
       --arrow-scale-km 40 --arrow-width-scale 0.4 \
       --firework --fw-trail --fw-core \
       --ray-length-mult 6.0 --ray-segments 8 --fw-expansion-km 440.0
   ```

3. **View the results:**
   ```bash
   cd outputs
   python3 -m http.server 8000
   # Open http://localhost:8000/weather_rays_len6_firework_tailfx.html
   ```

## 🎨 Visualization Features

### Core Weather Data
- **Temperature Mapping**: Blue-to-red color gradient representing temperature ranges
- **Wind Visualization**: Dynamic arrows showing wind direction and speed
- **Humidity Levels**: Affects marker sizing and atmospheric effects
- **Weather Conditions**: Symbol-coded weather states (clear, cloudy, rainy, etc.)

### Advanced Visual Effects
- **Radiating Rays**: Dynamic curves emanating from cities with undulating motion
- **Firework Bursts**: Explosive particle systems with trails and core flashes
- **Emission Animation**: Rays growing outward from city centers
- **Interactive Timeline**: Smooth transitions through 30 days of data

### Performance Modes
- **High Performance**: Optimized for smooth playback on standard hardware
- **Maximum Effects**: Full visual fidelity with all effects enabled
- **Lightweight**: Minimal effects for older systems or slow connections

## 📊 Technical Specifications

### Data Sources
- **Coverage**: 50+ major global cities
- **Time Range**: 30-day rolling window
- **Update Frequency**: Hourly data points
- **Parameters**: Temperature, humidity, wind speed/direction, weather conditions

### Technology Stack
- **Backend**: Python 3.7+ with Pandas, NumPy
- **Visualization**: Plotly.js with custom animations
- **Output Format**: Self-contained HTML5 with embedded data
- **Compatibility**: Modern browsers with WebGL support

### File Specifications
- **Output Size**: 50-100MB depending on configuration
- **Resolution**: Adaptive based on viewport
- **Performance**: 60fps capable on modern hardware
- **Interactivity**: Full mouse/touch support with detailed tooltips

## 🛠️ Configuration Options

The visualization system supports extensive customization:

- **Animation Speed**: 1-30 FPS
- **Visual Complexity**: Rays, particles, trails, undulation
- **Color Schemes**: Temperature-based gradients
- **Performance Tuning**: Quality vs. speed trade-offs
- **Time Granularity**: Daily or hourly data resolution

## 📈 Project Evolution

### Version 0919 (Initial Development)
- Basic weather mapping functionality
- Multiple visualization experiments
- Prototype animations and effects

### Version 0921 (Production Release)
- Enhanced performance and stability
- Comprehensive documentation
- Advanced visual effects
- Production-ready output

## 🎯 Learning Objectives

This project demonstrates proficiency in:

1. **Data Science**: Large dataset processing, time-series analysis
2. **Visualization**: Advanced plotting techniques, animation principles
3. **Web Technologies**: HTML5, JavaScript, responsive design
4. **Performance Optimization**: Efficient rendering, memory management
5. **User Experience**: Intuitive controls, accessibility considerations

## 🔗 Usage Examples

- **Academic Research**: Climate pattern analysis and presentation
- **Educational Tools**: Interactive geography and meteorology lessons
- **Data Journalism**: Weather story visualization for media
- **Business Intelligence**: Location-based weather impact analysis

## 🤝 Contributing

This project is part of coursework for SD5913. For questions or collaboration opportunities, please contact the author through the GitHub repository.

## 📄 License

This project is created for educational purposes as part of the SD5913 course curriculum.

---

**Note**: The generated HTML visualization file is large (94MB) due to embedded animation data. Consider using Git LFS for version control of large files in production environments.
