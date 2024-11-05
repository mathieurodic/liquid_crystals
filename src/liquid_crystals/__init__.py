"""
Liquid Crystals Simulator
========================

A package for simulating and visualizing liquid crystal behavior.
"""

from .lc import LC
from .fields import Uniform, InverseSquareDistance, InverseDistance, Angle

__version__ = "0.2.1"
__all__ = ['LC', 'Uniform', 'InverseSquareDistance', 'InverseDistance', 'Angle']