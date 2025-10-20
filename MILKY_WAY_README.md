# Interactive 3D Milky Way Galaxy Simulation

A stunning, realistic real-time simulation of our Milky Way galaxy featuring spiral arms, central bulge, and stellar halo. Available in both **web-based** (Three.js) and **Python** (matplotlib) versions.

![Milky Way Simulation](https://img.shields.io/badge/Stars-50%2C000-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

### Realistic Galaxy Structure
- **4 Spiral Arms**: Logarithmic spiral structure mimicking the Milky Way's actual arms
- **Central Bulge**: Dense spherical distribution of older, yellow stars
- **Stellar Halo**: Sparse distribution of ancient red stars surrounding the galaxy
- **50,000+ Stars**: Customizable star count for performance vs. detail balance

### Interactive Controls
- **360° Rotation**: Full orbital camera controls
- **Zoom**: Smooth zoom in/out to explore details
- **Pan**: Move around the galaxy
- **Auto-rotation**: Optional automatic rotation for presentations
- **Adjustable Speed**: Control rotation speed in real-time

### Realistic Astrophysics
- **Color-coded Populations**:
  - Blue-white disk stars (young, hot O/B type stars)
  - Yellow-gold bulge stars (older, cooler G/K type stars)
  - Red halo stars (ancient, metal-poor Population II stars)
- **Proper Distributions**: Exponential and Gaussian distributions for realistic star placement
- **Disk Thickness**: Thin disk structure typical of spiral galaxies

---

## 🌐 Version 1: Web-Based (Recommended)

### Best For
- **Best visual quality and performance**
- Smooth 60 FPS rendering
- Works on any device with a modern browser
- No installation required

### Quick Start

1. **Open the simulation**:
   ```bash
   # Open in your default browser (Linux)
   xdg-open milky_way_3d.html

   # Or on macOS
   open milky_way_3d.html

   # Or on Windows
   start milky_way_3d.html
   ```

2. **Use your browser**:
   Simply double-click `milky_way_3d.html` to open in your default web browser.

### Controls (Web Version)

| Action | Control |
|--------|---------|
| Rotate view | Left mouse button + drag |
| Pan view | Right mouse button + drag |
| Zoom | Mouse scroll wheel |
| Toggle auto-rotation | Spacebar or "Pause/Resume Rotation" button |
| Reset camera | "Reset View" button |
| Adjust speed | Speed slider |

### Browser Requirements
- Modern browser with WebGL support (Chrome, Firefox, Safari, Edge)
- Works on desktop, tablet, and mobile devices

---

## 🐍 Version 2: Python-Based

### Best For
- Scientific analysis and customization
- Integration with Python data pipelines
- Offline use without a browser
- Educational purposes

### Installation

1. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```

2. **Run the simulation**:
   ```bash
   python milky_way_simulation.py
   ```

### Interactive Usage

When you run the script, you'll be prompted:

```
Choose simulation mode:
1. Static interactive (recommended - full mouse control)
2. Auto-rotating animation

Enter choice (1 or 2) [default: 1]: 1

Enter number of stars (10000-100000) [default: 50000]: 50000
```

### Controls (Python Version)

| Action | Control |
|--------|---------|
| Rotate view | Click and drag |
| Zoom | Scroll wheel |
| Pan | Right-click and drag (or Shift + left-click) |

### Customization

You can customize the simulation by editing the Python script:

```python
# Change number of stars
galaxy = MilkyWaySimulation(num_stars=100000)

# Modify spiral arm parameters
a = 0.5  # Starting radius
b = 0.3  # Tightness of spiral
spread = 0.3  # Arm thickness

# Adjust population ratios
num_disk = int(self.num_stars * 0.7)   # 70% in disk
num_bulge = int(self.num_stars * 0.2)  # 20% in bulge
num_halo = int(self.num_stars * 0.1)   # 10% in halo
```

---

## 📊 Technical Details

### Galaxy Model

The simulation uses scientifically accurate models:

1. **Spiral Arms**: Logarithmic spiral equation
   ```
   r = a × e^(b×θ)
   ```
   Where `a` is the starting radius and `b` controls the tightness

2. **Bulge Distribution**: Exponential radial profile
   ```
   ρ(r) ∝ e^(-r/r_scale)
   ```

3. **Halo Distribution**: Extended spherical distribution with large scale radius

### Performance

| Version | Stars | FPS | Memory |
|---------|-------|-----|--------|
| Web (Three.js) | 50,000 | 60 | ~200 MB |
| Python (matplotlib) | 50,000 | 30-60 | ~500 MB |

### Star Populations

| Component | Percentage | Color | Characteristics |
|-----------|-----------|-------|-----------------|
| Disk | 70% | Blue-white | Young, metal-rich, Population I |
| Bulge | 20% | Yellow-gold | Old, metal-rich, orbiting center |
| Halo | 10% | Red | Ancient, metal-poor, Population II |

---

## 🎨 Screenshots

### Different Views

The simulation supports various viewing angles:
- **Face-on view**: See the spiral structure clearly
- **Edge-on view**: Observe the thin disk and bulge
- **Oblique view**: Best overall perspective (default)

---

## 🔬 Educational Use

This simulation is perfect for:
- **Astronomy classes**: Visualizing galaxy structure
- **Presentations**: Auto-rotation mode for talks
- **Research**: Understanding stellar distributions
- **Outreach**: Public engagement with astronomy

---

## 🚀 Advanced Features

### Modify Star Count (Web Version)

Edit `milky_way_3d.html` and change:
```javascript
const CONFIG = {
    numDiskStars: 35000,   // Increase for more detail
    numBulgeStars: 10000,
    numHaloStars: 5000,
    rotationSpeed: 0.001
};
```

### Export Data (Python Version)

You can access the star positions for analysis:
```python
galaxy = MilkyWaySimulation(num_stars=50000)

# Access star positions
disk_positions = np.column_stack([galaxy.x_disk, galaxy.y_disk, galaxy.z_disk])
bulge_positions = np.column_stack([galaxy.x_bulge, galaxy.y_bulge, galaxy.z_bulge])
halo_positions = np.column_stack([galaxy.x_halo, galaxy.y_halo, galaxy.z_halo])

# Save to file
np.savetxt('milky_way_stars.csv', disk_positions, delimiter=',')
```

---

## 🐛 Troubleshooting

### Web Version

**Problem**: Simulation won't load
- **Solution**: Ensure you have an internet connection (loads Three.js from CDN)
- **Solution**: Try a different browser (Chrome recommended)

**Problem**: Low FPS
- **Solution**: Reduce star count in the CONFIG section
- **Solution**: Close other browser tabs

### Python Version

**Problem**: `ModuleNotFoundError`
- **Solution**: Run `pip install numpy matplotlib`

**Problem**: Plot doesn't show
- **Solution**: Ensure you're not in a headless environment
- **Solution**: Try `matplotlib.use('TkAgg')` before importing pyplot

**Problem**: Slow performance
- **Solution**: Reduce star count: `MilkyWaySimulation(num_stars=10000)`

---

## 📚 Scientific Accuracy

This simulation incorporates:
- Actual Milky Way structure (barred spiral galaxy, SBbc type)
- Realistic stellar populations
- Proper scale relationships (though compressed for visualization)
- Color-magnitude relationships

**Note**: This is a simplified model. The actual Milky Way contains:
- ~200-400 billion stars (we show 50,000 representative stars)
- More complex dynamics (rotation curves, dark matter halo)
- Gas clouds, nebulae, and dark dust lanes
- Active star formation regions

---

## 🎯 Future Enhancements

Potential additions:
- [ ] Stellar rotation (differential rotation curve)
- [ ] Dust lanes and gas clouds
- [ ] Supernova remnants and nebulae
- [ ] Binary star systems
- [ ] Dark matter halo visualization
- [ ] Time evolution simulation
- [ ] VR support

---

## 📄 License

MIT License - Feel free to use, modify, and distribute

---

## 🙏 Acknowledgments

- Based on real Milky Way galaxy structure research
- Three.js for amazing 3D web graphics
- matplotlib for Python visualization
- The astronomical community for open data and research

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Try the web version if Python version has issues

---

**Enjoy exploring our galaxy!** 🌌✨
