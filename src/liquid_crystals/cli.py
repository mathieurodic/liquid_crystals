# liquid_crystals/cli.py
import click
import numpy as np
from liquid_crystals import LC

@click.command()
@click.option('--width', type=int, default=400, help='Width of the simulation grid')
@click.option('--height', type=int, default=400, help='Height of the simulation grid')
@click.option('--dtype', type=click.Choice(['float32', 'float64']), default='float32',
              help='Data type for calculations')
@click.option('--radius', type=int, default=2, help='Interaction radius for neighbor effects')
@click.option('--random-fields', type=int, default=10, help='Number of random fields to add')
@click.option('--threaded/--no-threaded', default=True, help='Run simulation in a separate thread')
@click.option('--steps', type=int, default=None, help='Number of simulation steps (None for infinite)')
@click.option('--neighbours-influence', type=float, default=0.01,
              help='Strength of neighbor interactions')
@click.option('--field-influence', type=float, default=0.8, help='Strength of field effects')
@click.option('--dt', type=float, default=1.0, help='Time step size')
@click.option('--viscosity-halftime', type=float, default=4000,
              help='Viscosity evolution parameter')
@click.option('--viscosity-start', type=float, default=0.01, help='Initial viscosity ratio')
@click.option('--viscosity-end', type=float, default=0.5, help='Final viscosity ratio')
@click.option('--cmap', type=str, default='special', help='Colormap for display')
def main(width, height, dtype, radius, random_fields, threaded, steps,
         neighbours_influence, field_influence, dt, viscosity_halftime,
         viscosity_start, viscosity_end, cmap):
    """Liquid Crystal Simulation CLI.
    
    This tool provides a command-line interface to run liquid crystal simulations
    with customizable parameters.
    """
    # Initialize simulation
    lc = LC(width=width, height=height, dtype=getattr(np, dtype), radius=radius)
    
    # Add random fields
    lc.add_random_fields(random_fields)
    
    # Run simulation
    lc.simulate(
        threaded=threaded,
        n=steps,
        neighbours_influence=neighbours_influence,
        field_influence=field_influence,
        dt=dt,
        viscosity_halftime=viscosity_halftime,
        viscosity_start=viscosity_start,
        viscosity_end=viscosity_end
    )
    
    # Display results
    lc.display(cmap=cmap)

if __name__ == '__main__':
    main()
