"""
Display Module for Liquid Crystal Simulation.

This module provides visualization capabilities for the liquid crystal simulation
using matplotlib. It supports both static and animated displays of the liquid
crystal state and field configurations.
"""

import matplotlib.animation
import matplotlib.pyplot
import matplotlib.colors
import scipy.ndimage
import numpy
from typing import Literal, List


class Display:
    """
    Display handler for liquid crystal simulation visualization.
    
    This class manages the visualization of liquid crystal states and fields
    using matplotlib. It supports both static displays and animations of the
    simulation progress.
    """

    def __init__(self, 
                 lc: 'LC',  # Forward reference to LC class
                 fps: int = 25,
                 cmap: Literal['twilight', 'baw', 'baw2', 'hsv', 'twilight_shifted', 'special'] = 'special',
                 show_field: bool = True,
                 resolution: int = 32):
        """
        Initialize the display handler.
        
        Args:
            lc: Liquid crystal simulation instance to visualize.
            fps (int, optional): Frames per second for animation. Defaults to 25.
            cmap (str, optional): Color map to use. Defaults to 'twilight'.
            resolution (int, optional): Grid resolution for field visualization. Defaults to 32.
        """
        self.fps = fps

        if cmap == 'baw':
            cmap_scale = 0.5 - 0.5 * numpy.cos(numpy.linspace(0, 2 * numpy.pi, 256))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'Cyclic grayscale', numpy.stack((cmap_scale,) * 3, axis=-1))
        elif cmap == 'baw2':
            cmap_scale = 0.5 - 0.5 * numpy.cos(numpy.linspace(0, 4 * numpy.pi, 512))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'Cyclic grayscale', numpy.stack((cmap_scale,) * 3, axis=-1))
        elif cmap == 'special':
            cmap_scale = []
            black = numpy.array((0,0,0), dtype=numpy.float32)
            gradient = 0.5 - 0.5 * numpy.cos(numpy.linspace(0, 2 * numpy.pi, 256))
            for color in [(1,0,0),(1,1,0),(0,1,0),(0,0,1)]:
                color = numpy.array(color, dtype=numpy.float32)
                cmap_color_scale = [value * color for value in gradient]
                cmap_scale += cmap_color_scale
            cmap_scale = cmap_scale[-128:] + cmap_scale[:-128]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'Special', cmap_scale)

        self.lc = lc
        self.grid_size = int(numpy.sqrt(lc.angles.shape[0] * lc.angles.shape[1]) / resolution)

        # Initialize matplotlib figure and axes
        self.fig, ax = matplotlib.pyplot.subplots()
        self.image = matplotlib.pyplot.imshow(lc.angles % numpy.pi, cmap=cmap, interpolation='none')

        # Setup colorbar
        colorbar = self.fig.colorbar(self.image, ticks=[.01 * numpy.pi, .5 * numpy.pi, .99 * numpy.pi])
        colorbar.ax.set_yticklabels(['0', 'π/2', 'π'])

        # Setup field visualization
        self.show_field = show_field
        if self.show_field:
            self._setup_field_visualization(resolution)

    def _setup_field_visualization(self, resolution):
        """Set up the field visualization quiver plot."""
        field_angles, field_intensities = self.lc.total_field
        
        # Create coordinate grids
        height, width = field_angles.shape
        x = numpy.arange(0, width, self.grid_size)
        y = numpy.arange(0, height, self.grid_size)
        X, Y = numpy.meshgrid(x, y)
        
        # Process field intensities
        field_intensities = scipy.ndimage.zoom(field_intensities, 1/self.grid_size, order=0)
        field_intensities = numpy.log(field_intensities) / numpy.log(10)
        field_intensities -= numpy.min(field_intensities)
        field_intensities /= numpy.max(field_intensities)

        # Sample field angles
        field_angles = field_angles[::self.grid_size, ::self.grid_size]
        field_angles = field_angles[:field_intensities.shape[0], :field_intensities.shape[1]]

        # Prepare quiver plot data
        X = X[:field_intensities.shape[0], :field_intensities.shape[1]]
        Y = Y[:field_intensities.shape[0], :field_intensities.shape[1]]
        U = numpy.cos(field_angles)
        V = numpy.sin(field_angles)
        C = (255 * field_intensities).astype(numpy.int8)
        
        # Create quiver plot
        self.quiver = matplotlib.pyplot.quiver(
            X, Y, U, V, C,
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom', 
                [(0.5,0.5,0.5,0.8), (0.5,0.5,0.5,0.1)], 
                N=256
            ),
            scale=resolution,
            pivot='middle',
            headlength=0,
            width=.003,
            headwidth=.003)

    def animate(self, i: int) -> List:
        """
        Animation update function.
        
        Args:
            i (int): Frame index.
            
        Returns:
            list: List of artists to update.
        """
        self.image.set_array(self.lc.angles % numpy.pi)
        return [self.image, self.quiver] if self.show_field else [self.image]
        
    def start(self, animate: bool = True):
        """
        Start the display.
        
        Args:
            animate (bool, optional): Whether to animate the display. Defaults to True.
        """
        if animate:
            anim = matplotlib.animation.FuncAnimation(
                self.fig, 
                self.animate, 
                interval=1000/self.fps,
                blit=True, 
                cache_frame_data=False
            )
            
        # Maximize window based on backend
        backend_name = matplotlib.pyplot.get_backend().lower()
        figure_manager = matplotlib.pyplot.get_current_fig_manager()
        if backend_name == 'tkagg':
            figure_manager.resize(*figure_manager.window.maxsize())
        elif backend_name == 'wxagg':
            figure_manager.frame.Maximize(True)
        else:  # Qt4Agg, Qt5Agg, etc.
            figure_manager.window.showMaximized()
            
        matplotlib.pyplot.show()
