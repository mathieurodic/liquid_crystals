"""
Field Generation Module for Liquid Crystal Simulation.

This module provides functions to generate various types of fields that influence
the liquid crystal behavior. Each field generator is a function that returns
another function calculating field values at given coordinates.
"""

import numpy as np
from typing import Callable, Tuple

# Type alias for field functions
FieldFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

def Uniform(value: float) -> FieldFunction:
    """
    Create a uniform field with constant value everywhere.
    
    Args:
        value (float): The constant field value to be applied uniformly.
    
    Returns:
        callable: A function that returns the uniform field value for any coordinates.
    
    Example:
        >>> field_func = Uniform(1.0)
        >>> result = field_func(i_coords, j_coords)  # Returns array of 1.0s
    """
    def function(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return value
    return function
    
def InverseSquareDistance(center_i: int, center_j: int, value: float) -> FieldFunction:
    """
    Create a field that follows inverse square law from a central point.
    
    Args:
        center_i (int): The i-coordinate of the field center.
        center_j (int): The j-coordinate of the field center.
        value (float): The field strength at the center point.
    
    Returns:
        callable: A function that calculates field values based on inverse square distance.
    
    Example:
        >>> field_func = InverseSquareDistance(10, 10, 2.0)
        >>> result = field_func(i_coords, j_coords)
    """
    def function(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        di = i - center_i
        dj = j - center_j
        try:
            result = value / (di * di + dj * dj + 1e-9)
        except RuntimeWarning:
            pass
        result[center_i][center_j] = value
        return result
    return function
    
def InverseDistance(center_i: int, center_j: int, value: float) -> FieldFunction:
    """
    Create a field that follows inverse distance law from a central point.
    
    Args:
        center_i (int): The i-coordinate of the field center.
        center_j (int): The j-coordinate of the field center.
        value (float): The field strength at the center point.
    
    Returns:
        callable: A function that calculates field values based on inverse distance.
    
    Example:
        >>> field_func = InverseDistance(10, 10, 2.0)
        >>> result = field_func(i_coords, j_coords)
    """
    def function(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        di = i - center_i
        dj = j - center_j
        result = value / np.hypot(di, dj)
        result[center_i][center_j] = value
        return result
    return function

def Angle(center_i: int, center_j: int, constant: float = 0.0, factor: float = 1) -> FieldFunction:
    """
    Create a field where values are angles relative to a central point.
    
    Args:
        center_i (int): The i-coordinate of the field center.
        center_j (int): The j-coordinate of the field center.
        constant (float, optional): Constant angle offset. Defaults to 0.0.
        factor (float, optional): Angle scaling factor. Defaults to 1.
    
    Returns:
        callable: A function that calculates angular field values.
    
    Example:
        >>> field_func = Angle(10, 10, constant=np.pi/4)
        >>> result = field_func(i_coords, j_coords)
    """
    def function(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        di = i - center_i
        dj = j - center_j
        return constant + factor * np.angle(di + 1j * dj)
    return function
