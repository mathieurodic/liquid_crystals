"""
Liquid Crystal Simulation Module.

This module implements the core liquid crystal simulation functionality,
modeling the behavior of liquid crystal molecules under various field conditions
and neighbor interactions.
"""

import numpy
import scipy.signal
import threading
import random
import inspect
from typing import Literal, Callable, Optional

from .display import Display
from . import fields


class LC:
    """
    Liquid Crystal Simulation Class.
    
    This class implements the core simulation of liquid crystal behavior,
    handling molecular orientations, field effects, and neighbor interactions.
    """

    def __init__(self, width, height, dtype=numpy.float32, radius=1):
        """
        Initialize liquid crystal simulation.
        
        Args:
            width (int): Width of the simulation grid.
            height (int): Height of the simulation grid.
            dtype (numpy.dtype, optional): Data type for calculations. Defaults to numpy.float32.
            radius (int, optional): Interaction radius for neighbor effects. Defaults to 1.
        """
        self.fields = []
        self.angles = numpy.random.uniform(low = 0.0,
                                           high = 2.0 * numpy.pi,
                                           size = (height, width),
                                           ).astype(dtype)
        self.speeds = numpy.zeros(shape=(height, width), dtype=dtype)
        
        # Setup acceleration filter
        size = 2 * radius + 1
        self.accelerations_filter = numpy.zeros((size, size), dtype=dtype)
        for i in range(size):
            for j in range(size):
                if i == radius and j == radius:
                    continue
                di = i - radius
                dj = j - radius
                self.accelerations_filter[i][j] = 1.0 / (di * di + dj * dj)
        self.accelerations_filter /= numpy.sum(self.accelerations_filter)
        self.accelerations_filter[radius][radius] = -1.0
        self.is_running = False
        self._simulate_total_field()

    def add_field(self, 
                 angle_decorated_function: Callable,
                 strength_decorated_function: Callable,
                 simulate_total: bool = True,
                 **kwargs):
        """
        Add a new field to the simulation.
        
        Args:
            angle_decorated_function (callable): Function generating field angles.
            strength_decorated_function (callable): Function generating field strengths.
            simulate_total (bool, optional): Whether to update total field. Defaults to True.
            **kwargs: Additional arguments passed to field functions.
        """
        self.fields.append(tuple(
            decorated_function(**{
                k: v
                for k, v in kwargs.items()
                if k in inspect.getfullargspec(decorated_function).args
            })
            for decorated_function in (angle_decorated_function, strength_decorated_function)
        ))
        self._simulate_total_field()

    def add_random_fields(self, count):
        """
        Add multiple random fields to the simulation.
        
        Args:
            count (int): Number of random fields to add.
        """
        for i in range(count):
            angle_c = numpy.pi * random.random()
            angle_f = random.randrange(-1, 3)
            center_i = random.randrange(self.angles.shape[0])
            center_j = random.randrange(self.angles.shape[1])
            intensity = 4. * random.random()
            self.add_field(
                fields.Angle,
                fields.InverseSquareDistance,
                constant=angle_c,
                factor=angle_f,
                center_i=center_i, center_j=center_j, value=intensity,
                simulate_total=False)
        self._simulate_total_field()

    def _simulate_total_field(self):
        """Calculate the total field from all individual fields."""
        shape, dtype = self.angles.shape, self.angles.dtype
        indices = numpy.indices(shape, dtype)
        result = numpy.zeros(shape, numpy.complex64)
        for angle_function, strength_function in self.fields:
            angle = angle_function(indices[0], indices[1])
            strength = strength_function(indices[0], indices[1])
            result += strength * numpy.exp(1j * angle)
        result_angle = numpy.angle(result).astype(dtype)
        result_angle[result_angle < 0] += numpy.pi
        self.total_field = result_angle, numpy.absolute(result).astype(dtype)

    def simulate(self, 
                threaded: bool,
                n: Optional[int] = None,
                neighbours_influence: float = 0.05,
                field_influence: float = 0.8,
                dt: float = 1.0,
                viscosity_halftime: float = 5000,
                viscosity_start: float = 0.01):
        """
        Run the simulation.
        
        Args:
            threaded (bool): Whether to run in a separate thread.
            n (int, optional): Number of simulation steps. Defaults to None (infinite).
            neighbours_influence (float, optional): Strength of neighbor interactions. Defaults to 0.05.
            field_influence (float, optional): Strength of field effects. Defaults to 0.8.
            dt (float, optional): Time step size. Defaults to 1.0.
            viscosity_halftime (float, optional): Viscosity evolution parameter. Defaults to 5000.
            viscosity_start (float, optional): Initial viscosity ratio. Defaults to 0.01.
            
        Returns:
            threading.Thread, optional: If threaded is True, returns the simulation thread.
        """
        if threaded:
            thread = threading.Thread(target=self.simulate, args=(False, n, neighbours_influence, field_influence, dt, viscosity_halftime, viscosity_start))
            thread.start()
            return thread
        self.is_running = True
        if n is None:
            n = (2 << 32)  # Very large number for "infinite" simulation
        for index in range(n):
            if not self.is_running:
                break
            print(f'simulate {index}', end='\r')
            self._simulate_step(index=index, neighbours_influence=neighbours_influence,
                               field_influence=field_influence, dt=dt,
                               viscosity_halftime=viscosity_halftime, viscosity_start=viscosity_start)
        print()

    def _simulate_step(self,
                      index: int,
                      neighbours_influence: float,
                      field_influence: float,
                      dt: float,
                      viscosity_halftime: float,
                      viscosity_start: float):
        """
        Perform a single simulation step.
        
        Args:
            index (int): Current simulation step index.
            neighbours_influence (float): Strength of neighbor interactions.
            field_influence (float): Strength of field effects.
            dt (float): Time step size.
            viscosity_halftime (float): Viscosity evolution parameter.
            viscosity_start (float): Initial viscosity ratio.
        """
        # Calculate viscosity
        viscosity_max = 1.0 / dt
        viscosity_min = viscosity_max * viscosity_start
        viscosity = viscosity_min + viscosity_max * (index / viscosity_halftime) / (1.0 + index / viscosity_halftime)

        # Calculate neighbor effects
        neighbours_angles_deltas = scipy.signal.convolve2d(
            in1=self.angles,
            in2=self.accelerations_filter,
            mode='same'
        )
        neighbours_accelerations = numpy.sin(2.0 * neighbours_angles_deltas)

        # Calculate field effects
        field_angles_deltas = self.total_field[0] - self.angles
        field_accelerations = self.total_field[1] * numpy.sin(2.0 * field_angles_deltas)
        
        # Combine all effects
        accelerations = neighbours_influence * neighbours_accelerations + field_influence * field_accelerations
        accelerations -= viscosity * self.speeds

        # Update speeds and angles
        self.speeds += accelerations * dt
        self.speeds[numpy.abs(self.speeds) > (numpy.pi / 2) / dt] /= 2.0
        self.angles += self.speeds * dt
        self.angles %= numpy.pi
        
        # Apply boundary conditions
        self.angles[0:2, 0:2] = numpy.pi / 2
        self.angles[0] = self.angles[-1] = 0
        self.angles[..., 0] = self.angles[..., -1] = numpy.pi / 2
        self.speeds[0] = self.speeds[-1] = self.speeds[..., 0] = self.speeds[..., -1] = 0
    
    def display(self,
               animate: bool = True,
               stop_computing_afterwards: bool = True,
               fps: int = 25,
               scale: float = 1,
               cmap: Literal['twilight', 'baw', 'hsv', 'twilight_shifted'] = 'twilight',
               resolution: int = 32):
        """
        Display the simulation state.
        
        Args:
            animate (bool, optional): Whether to animate the display. Defaults to True.
            stop_computing_afterwards (bool, optional): Whether to stop computing when display gets claused. Defaults to True.
            fps (int, optional): Frames per second for animation. Defaults to 25.
            scale (float, optional): Display scale factor. Defaults to 1.
            cmap (str, optional): Color map to use. Defaults to 'twilight'.
            resolution (int, optional): Grid resolution for field visualization. Defaults to 32.
        """
        display = Display(
            lc=self,
            fps=fps,
            scale=scale,
            cmap=cmap,
            resolution=resolution
        )
        display.start(animate=animate)
        if stop_computing_afterwards:
            self.is_running = False

