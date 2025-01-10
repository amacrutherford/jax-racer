""" 
Drawn from: https://github.com/facebookresearch/dcd/blob/main/envs/box2d/car_racing_bezier.py

"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from bezier import get_random_points_fixed, get_bezier_curve_fixed

class CarRacing:
    
    def __init__(self,):
        
        pass
    
    def create_track(self, rng):
        
        MAX_NUM_CHECKPOINTS = 12
        NUM_CHECKPOINTS = 10
        NUM_POINTS = 30
        TRACK_WIDTH = 1
        
        a = get_random_points_fixed(rng, NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS)
        x, y, _ = get_bezier_curve_fixed(a, NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS, NUM_POINTS)
        print('x:', x.shape)
                
        points = jnp.column_stack([x, y])
        
        print(points.shape)
        
        def _calc_v(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            alpha = jnp.arctan2(dy, dx)
            beta = alpha + jnp.pi / 2.0
            
            return alpha, beta, x1, y1 
        
        track = jax.vmap(_calc_v)(points[:-1], points[1:])
        
        min_x, max_x = jnp.min(x), jnp.max(x)
        min_y, max_y = jnp.min(y), jnp.max(y)
        
        x_offset = (min_x + max_x) / 2.0
        y_offset = (min_y + max_y) / 2.0
        
        def _create_tiles(t1, t2):
            alpha1, beta1, x1, y1 = t1

            alpha2, beta2, x2, y2 = t2
            
            road1_l = (
                x1 - TRACK_WIDTH * jnp.cos(beta1) - x_offset,
                y1 - TRACK_WIDTH * jnp.sin(beta1) - y_offset,
            )
            road1_r = (
                x1 + TRACK_WIDTH * jnp.cos(beta1) - x_offset,
                y1 + TRACK_WIDTH * jnp.sin(beta1) - y_offset,
            )
            road2_l = (
                x2 - TRACK_WIDTH * jnp.cos(beta2) - x_offset,
                y2 - TRACK_WIDTH * jnp.sin(beta2) - y_offset,
            )
            road2_r = (
                x2 + TRACK_WIDTH * jnp.cos(beta2) - x_offset,
                y2 + TRACK_WIDTH * jnp.sin(beta2) - y_offset,
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            
        
        
if __name__ == '__main__':
    
    cr = CarRacing()
    cr.create_track(jax.random.PRNGKey(0))      