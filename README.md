# Liquid Crystals Simulator

A Python package for simulating and visualizing liquid crystal behavior under various field conditions.

![Image](./screenshot.jpg)

## Features

- Simulate liquid crystal molecular orientation dynamics
- Visualize the simulation in real-time
- Add custom fields with different geometries
- Configurable simulation parameters
- Interactive display with matplotlib

## Installation

```bash
pip install liquid-crystals
```

## Quick Start

```python
from liquid_crystals import LC

# Create a liquid crystal simulation
lc = LC(width=400, height=400, radius=2)

# Add random fields
lc.add_random_fields(10)

# Start simulation in a separate thread
lc.simulate(threaded=True)

# Display the simulation
lc.display()
```

## License

MIT License
