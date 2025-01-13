""" 
Drawn from: https://github.com/facebookresearch/dcd/blob/main/envs/box2d/car_racing_bezier.py

"""

import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct
import matplotlib.pyplot as plt
from functools import partial

from bezier import get_random_points_fixed, get_bezier_curve_fixed

from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.maths import rmat, cw
from jax2d.scene import (
    add_circle_to_scene,
    add_rectangle_to_scene,
    add_fixed_joint_to_scene,
    add_revolute_joint_to_scene,
    add_thruster_to_scene,
    add_polygon_to_scene,
)
from jax2d.sim_state import StaticSimParams, SimParams, SimState

from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_circle,
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

import pygame

SCREEN_SIZE = (500, 500)
BKGD_COLOR = jnp.array([0.0, 0.0, 0.0])

class CarRacing:
    
    def __init__(self,
                 max_num_checkpoints: int=10,
                 num_points: int=5,
                 track_width=0.05,):
        
        self.max_num_checkpoints = max_num_checkpoints
        self.num_points = num_points
        self.track_width = track_width
        
        self.static_sim_params = self.get_static_sim_params()
        self.sim_params = SimParams()
        self.engine = PhysicsEngine(self.static_sim_params)
        
        
    def get_static_sim_params(self) -> StaticSimParams:
        
        static_sim_params = StaticSimParams(
            num_polygons=self.max_num_checkpoints*self.num_points+1,
            num_circles=2,
            max_polygon_vertices=4,
            num_static_fixated_polys=self.max_num_checkpoints*self.num_points,
        )
        return static_sim_params
    
    def reset(self, rng: chex.PRNGKey) -> SimState:
        
        static_sim_params = self.get_static_sim_params()
        
        sim_state = create_empty_sim(static_sim_params, add_floor=False, add_walls_and_ceiling=False)
        sim_state = self.create_track(rng, sim_state, static_sim_params)
        
        sim_state = self.add_car(sim_state, static_sim_params)
        sim_state = self.add_car(sim_state, static_sim_params)
        sim_state = self.add_car(sim_state, static_sim_params)
        print('sim_state:', sim_state.polygon.position)
        
        return sim_state
    
    # @partial(jax.jit, static_argnames=("self",))
    def step(self, sim_state: SimState):
        
        empty_act = jnp.zeros((self.static_sim_params.num_joints + self.static_sim_params.num_thrusters,))
        
        sim_state, (rr_manifolds, _, _) = self.engine.step(sim_state, self.sim_params, empty_act)
        print('rr_manifolds:', rr_manifolds.active)
        # raise
        
        return sim_state
    
    def create_track(self, rng: chex.PRNGKey, sim_state: SimState, static_sim_params: StaticSimParams):
        
        NUM_CHECKPOINTS = 9    
        
        a = get_random_points_fixed(rng, NUM_CHECKPOINTS, self.max_num_checkpoints)
        # print('a:', a.shape, a)  # between 0 and 1
        x, y, _ = get_bezier_curve_fixed(a, NUM_CHECKPOINTS, self.max_num_checkpoints, self.num_points)
        # print('x:', x.shape, x)  # also between 0 and 1
                
        fig, ax = plt.subplots()
        plt.plot(x, y)
        plt.savefig('track.png')
        plt.close()

        points = jnp.column_stack([x, y])
        
        # print(points.shape)
        
        def _calc_v(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            alpha = jnp.arctan2(dy, dx)
            beta = alpha + jnp.pi / 2.0
            
            return jnp.array([alpha, beta, x1, y1])
        
        track = jax.vmap(_calc_v)(points[:-1], points[1:])
        # print('track:', track.shape, track)
        
        min_x, max_x = jnp.min(x), jnp.max(x)
        min_y, max_y = jnp.min(y), jnp.max(y)
        
        x_offset = (min_x + max_x) / 2.0
        y_offset = (min_y + max_y) / 2.0
        
        def _create_tiles(sim_state: SimState, t1_t2):
            t1, t2 = t1_t2
            alpha1, beta1, x1, y1 = t1
            alpha2, beta2, x2, y2 = t2
            
            road1_l = (
                x1 - self.track_width * jnp.cos(beta1), #- x_offset,
                y1 - self.track_width * jnp.sin(beta1), #- y_offset,
            )
            road1_r = (
                x1 + self.track_width * jnp.cos(beta1), #- x_offset,
                y1 + self.track_width * jnp.sin(beta1), #- y_offset,
            )
            road2_l = (
                x2 - self.track_width * jnp.cos(beta2), #- x_offset,
                y2 - self.track_width * jnp.sin(beta2), #- y_offset,
            )
            road2_r = (
                x2 + self.track_width * jnp.cos(beta2), #- x_offset,
                y2 + self.track_width * jnp.sin(beta2), #- y_offset,
            )
            vertices = jnp.array([road1_l, road1_r, road2_r, road2_l])
            # print('vertices:', vertices)
            vertices = cw(vertices)
            # print('vertices:', vertices)
            # raise
            # need to sort ccw
            
            center_pos = jnp.mean(vertices, axis=0)
            vertices = vertices - center_pos
            
            sim_state, (poly_idx, global_idx) = add_polygon_to_scene(
                sim_state,
                static_sim_params,
                center_pos,
                vertices,
                n_vertices=4,
                fixated=True,
            )
            
            return sim_state, (poly_idx, global_idx)
        
        t1_t2 = (track[:-1], track[1:])
        
        sim_state, idxs = jax.lax.scan(_create_tiles, sim_state, t1_t2)
        
        return sim_state
    
    def add_car(self, sim_state: SimState, static_sim_params: StaticSimParams):
        
        sim_state, (_, r_index) = add_rectangle_to_scene(
            sim_state, static_sim_params, position=jnp.array([1.25, 0.8]), dimensions=jnp.array([1.0, 0.4])
        )
        
        return sim_state
    
def make_track_renderer(static_sim_params):
    SCALE = 1
    ppud = 100
    patch_size = 1024
    screen_padding = patch_size
    full_screen_size = (
        SCREEN_SIZE[0] + 2 * screen_padding,
        SCREEN_SIZE[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * ppud + screen_padding

    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

    polygon_shader = add_mask_to_shader(make_fragment_shader_convex_dynamic_ngon_with_edges(4))
    quad_renderer = make_renderer(full_screen_size, polygon_shader, (patch_size, patch_size), batched=True)
    
    @jax.jit
    def render_track(sim_state: SimState):
        
        pixels = cleared_screen

        # Rectangles
        rect_positions_pixel_space = _world_space_to_pixel_space(sim_state.polygon.position*SCALE)
        rectangle_rmats = jax.vmap(rmat)(sim_state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :],
            repeats=static_sim_params.max_polygon_vertices,
            axis=1,
        )
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            sim_state.polygon.position[:, None, :]*SCALE + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, sim_state.polygon.vertices*SCALE)
        )
        rect_patch_positions = (rect_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)

        rect_colours = jnp.ones((static_sim_params.num_polygons, 3)) * 128.0
        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            sim_state.polygon.n_vertices,
            sim_state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)
        
        # Crop out the sides
        return pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]

    return render_track
        
        
if __name__ == '__main__':
    debug = False
    
    cr = CarRacing()
    
    static_sim_params = cr.get_static_sim_params()
    # NOTE need to increase number of polygons
    # sim_params = SimParams()
    # engine = PhysicsEngine(static_sim_params)

    # Create scene
    # sim_state = create_empty_sim(static_sim_params, add_floor=False, add_walls_and_ceiling=False)
    
    with jax.disable_jit(disable=debug):
        sim_state = cr.reset(jax.random.PRNGKey(1))
        
        cr.step(sim_state)
        
        # sim_state = cr.create_track(jax.random.PRNGKey(1), sim_state, static_sim_params)
    
    # print(sim_state.polygon)
    
    # plot positions
    fig, ax = plt.subplots()
    poly_centers = sim_state.polygon.position
    ax.plot(poly_centers[:, 0], poly_centers[:, 1], 'o')
    
    # plot vertices
    for i in range(sim_state.polygon.position.shape[0]):
        print('position:', sim_state.polygon.position[i])
        vertices = sim_state.polygon.vertices[i] + sim_state.polygon.position[i]
        vertices = jnp.vstack([vertices, vertices[0]])
        ax.plot(vertices[:, 0], vertices[:, 1], '+', color='red')
        break 
    plt.savefig('poly_centers.png')
    
    render_track = make_track_renderer(static_sim_params)
    
    pixels = render_track(sim_state)
    
    pygame.init()
    screen_surface = pygame.display.set_mode(SCREEN_SIZE)
    
    surface = pygame.surfarray.make_surface(np.array(pixels)[:, ::-1])
    screen_surface.blit(surface, (0, 0))
    pygame.display.flip()
    # save pygame display to png
    pygame.image.save(screen_surface, 'track_pygame.png')
    