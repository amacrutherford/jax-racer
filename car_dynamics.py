"""
Drawn from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_dynamics.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from functools import partial
import chex
import pygame
from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_circle,
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

from jax2d.collision import resolve_collision
from jax2d.engine import PhysicsEngine, create_empty_sim, create_empty_car_sim
from jax2d.maths import rmat
from jax2d.scene import (
    add_circle_to_scene,
    add_rectangle_to_scene,
    add_fixed_joint_to_scene,
    add_revolute_joint_to_scene,
    add_thruster_to_scene,
    add_polygon_to_scene,
)
from jax2d.sim_state import StaticSimParams, SimParams, SimState, CollisionManifold

from bezier import get_random_points_fixed, get_bezier_curve_fixed, cw_sort


FORWARD = jnp.array([0, 1])
SIDE = jnp.array([1, 0])

ENGINE_POWER = 4e4
WHEEL_MOMENT_OF_INERTIA = 1.6
FRICTION_LIMIT = 400

SCREEN_DIM = (500, 500)

def make_render_pixels(static_sim_params, screen_dim):
    ppud = 100
    patch_size = 512
    screen_padding = patch_size
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * ppud + screen_padding

    clear_colour = jnp.array([19.0, 133.0, 16.0])
    cleared_screen = clear_screen(full_screen_size, clear_colour)

    circle_shader = add_mask_to_shader(fragment_shader_circle)
    circle_renderer = make_renderer(full_screen_size, circle_shader, (patch_size, patch_size), batched=True)

    polygon_shader = add_mask_to_shader(make_fragment_shader_convex_dynamic_ngon_with_edges(4))
    quad_renderer = make_renderer(full_screen_size, polygon_shader, (patch_size, patch_size), batched=True)

    @jax.jit
    def render_pixels(state):
        pixels = cleared_screen

        # Rectangles
        rect_positions_pixel_space = _world_space_to_pixel_space(state.polygon.position)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :],
            repeats=static_sim_params.max_polygon_vertices,
            axis=1,
        )
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rect_patch_positions = (rect_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)

        rect_colours = jnp.zeros((static_sim_params.num_polygons, 3))
        rect_colours = rect_colours.at[4, 0].set(255.0)
        rect_colours = rect_colours.at[5, :].set(255.0)
        rect_colours = rect_colours.at[8, :].set(255.0)

        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            state.polygon.n_vertices,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = (circle_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
        circle_patch_positions = jnp.maximum(circle_patch_positions, 0)

        circle_colours = jnp.ones((static_sim_params.num_circles, 3)) * 255.0

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Crop out the sides
        return pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]

    return render_pixels

@struct.dataclass
class RacerState(SimState):
    track: jnp.ndarray = jnp.zeros((12, 2))

CAR_DIMS = jnp.array([0.15, 0.3])
CAR_HALF_DIMS = CAR_DIMS / 2.0
CAR_POS = jnp.array([2.5, 2.5])
WHEEL_DIMS = jnp.array([0.04, 0.08])

class Car:
    def __init__(self,
                 static_sim_params: StaticSimParams,
                 max_num_checkpoints: int = 12,
                 num_points_per_checkpoint: int = 5,
                 track_width: float = 1.0,):
        
        self.static_sim_params = static_sim_params
        self.engine = PhysicsEngine(self.static_sim_params)
        self.dt = 0.01
        self.step_fn = jax.jit(self.engine.step)

        self.max_num_checkpoints = max_num_checkpoints
        self.num_points_per_checkpoint = num_points_per_checkpoint
        self.track_width = track_width

        sim_state = create_empty_sim(self.static_sim_params, floor_offset=0.0)
        sim_state = sim_state.replace(gravity=jnp.zeros(2))

        # add rectangle for car body
        sim_state, (_, self.r_index) = add_rectangle_to_scene(
            sim_state, self.static_sim_params, position=CAR_POS, dimensions=CAR_DIMS
        )

        sim_state, (_, self.w0_index) = add_rectangle_to_scene(
            sim_state,
            self.static_sim_params,
            position=CAR_POS + CAR_HALF_DIMS * jnp.array([1.0, 1.0]),
            dimensions=WHEEL_DIMS,
            friction=0.0,
        )

        sim_state, (_, self.w1_index) = add_rectangle_to_scene(
            sim_state,
            self.static_sim_params,
            position=CAR_POS + CAR_HALF_DIMS * jnp.array([1.0, -1.0]),
            dimensions=WHEEL_DIMS,
            friction=0.0,
        )

        sim_state, (_, self.w2_index) = add_rectangle_to_scene(
            sim_state,
            self.static_sim_params,
            position=CAR_POS + CAR_HALF_DIMS * jnp.array([-1.0, -1.0]),
            dimensions=WHEEL_DIMS,
            friction=0.0,
        )

        sim_state, (_, self.w3_index) = add_rectangle_to_scene(
            sim_state,
            self.static_sim_params,
            position=CAR_POS + CAR_HALF_DIMS * jnp.array([-1.0, 1.0]),
            dimensions=WHEEL_DIMS,
            friction=0.0,
        )

        sim_state, self.w0_j_index = add_fixed_joint_to_scene(
            sim_state,
            self.static_sim_params,
            a_index=self.r_index,
            b_index=self.w0_index,
            a_relative_pos=CAR_HALF_DIMS * jnp.array([1.0, 1.0]),
            b_relative_pos=jnp.zeros(2),
        )

        sim_state, self.w1_j_index = add_fixed_joint_to_scene(
            sim_state,
            self.static_sim_params,
            a_index=self.r_index,
            b_index=self.w1_index,
            a_relative_pos=CAR_HALF_DIMS * jnp.array([1.0, -1.0]),
            b_relative_pos=jnp.zeros(2),
        )

        sim_state, self.w2_j_index = add_fixed_joint_to_scene(
            sim_state,
            self.static_sim_params,
            a_index=self.r_index,
            b_index=self.w2_index,
            a_relative_pos=CAR_HALF_DIMS * jnp.array([-1.0, -1.0]),
            b_relative_pos=jnp.zeros(2),
        )

        sim_state, self.w3_j_index = add_fixed_joint_to_scene(
            sim_state,
            self.static_sim_params,
            a_index=self.r_index,
            b_index=self.w3_index,
            a_relative_pos=CAR_HALF_DIMS * jnp.array([-1.0, 1.0]),
            b_relative_pos=jnp.zeros(2),
        )

        self.init_sim_state = sim_state
    
    def reset(self, rng: chex.PRNGKey):

        # Create scene
        
        
        # fold self.init_sim_state into sim_state, below is ugly but **kwargs doens't work..
        car_state = RacerState(
            polygon=self.init_sim_state.polygon,
            circle=self.init_sim_state.circle,
            joint=self.init_sim_state.joint,
            thruster=self.init_sim_state.thruster,
            collision_matrix=self.init_sim_state.collision_matrix,
            acc_rr_manifolds=self.init_sim_state.acc_rr_manifolds,
            acc_cr_manifolds=self.init_sim_state.acc_cr_manifolds,
            acc_cc_manifolds=self.init_sim_state.acc_cc_manifolds,
            gravity=self.init_sim_state.gravity,
        )

        car_state = self.create_track(rng, car_state, self.static_sim_params)

        return car_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, sim_state: RacerState, sim_params: SimParams, action: jnp.ndarray):

        actions = -jnp.ones(self.static_sim_params.num_joints + self.static_sim_params.num_thrusters)
        sim_state, _ = self.step_fn(sim_state, sim_params, actions)

        # Steer wheels
        steered_state = self._steer(sim_state, action[0])
        no_steer_state = self._return_wheels_to_centre(sim_state)
        sim_state = jax.tree.map(
            lambda x, y: jax.lax.select(action[0] != 0.0, x, y), steered_state, no_steer_state
        )

        # Cap wheel rotations
        sim_state = self._clip_wheel_rotation(sim_state, self.w0_index, self.w0_j_index)
        sim_state = self._clip_wheel_rotation(sim_state, self.w3_index, self.w3_j_index)
        # gas_act = 0.2  # (0, 1)
        # gas_act = jnp.clip(gas_act, 0.0, 1.0)

        gas_state = self._apply_gas(sim_state, action[1])
        sim_state = jax.tree.map(
            lambda x, y: jax.lax.select(action[1] != 0.0, x, y), gas_state, sim_state
        )

        brake_sim_state = self._apply_brake(sim_state, self.w0_index)
        brake_sim_state = self._apply_brake(brake_sim_state, self.w1_index)
        brake_sim_state = self._apply_brake(brake_sim_state, self.w2_index)
        brake_sim_state = self._apply_brake(brake_sim_state, self.w3_index)
        
        sim_state = jax.tree.map(
            lambda x, y: jax.lax.select(action[2] > 0.0, x, y), brake_sim_state, sim_state
        )

        sim_state = self._apply_wheel_impulse(sim_state, self.w0_index)
        sim_state = self._apply_wheel_impulse(sim_state, self.w1_index)
        sim_state = self._apply_wheel_impulse(sim_state, self.w2_index)
        sim_state = self._apply_wheel_impulse(sim_state, self.w3_index)

        # Cancel out lateral wheel movement
        sim_state = self._apply_wheel_lateral_impulse(sim_state, sim_params, self.w0_index)
        sim_state = self._apply_wheel_lateral_impulse(sim_state, sim_params, self.w1_index)
        sim_state = self._apply_wheel_lateral_impulse(sim_state, sim_params, self.w2_index)
        sim_state = self._apply_wheel_lateral_impulse(sim_state, sim_params, self.w3_index)

        return sim_state

    def _steer(self, sim_state: SimState, wheel_steer: float):
        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(
                rotation=sim_state.polygon.rotation.at[self.w0_index].add(wheel_steer)
            )
        )
        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(
                rotation=sim_state.polygon.rotation.at[self.w3_index].add(wheel_steer)
            )
        )
        return sim_state

    def _return_wheels_to_centre(self, sim_state: SimState):
        r0 = sim_state.polygon.rotation[self.w0_index] - sim_state.polygon.rotation[self.r_index]
        r0 *= 0.9
        r0 = sim_state.polygon.rotation[self.r_index] + r0

        r3 = sim_state.polygon.rotation[self.w3_index] - sim_state.polygon.rotation[self.r_index]
        r3 *= 0.9
        r3 = sim_state.polygon.rotation[self.r_index] + r3

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(
                rotation=sim_state.polygon.rotation.at[self.w0_index].set(r0).at[self.w3_index].set(r3)
            )
        )
        return sim_state
    
    def _apply_gas(self, sim_state: SimState, d_omega: float) -> SimState:
        # TODO clip d_omega value
        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(
                radius=sim_state.polygon.radius.at[self.w1_index].add(d_omega).at[self.w2_index].add(d_omega)
            )
        )
        return sim_state

    def _apply_brake(self, sim_state: SimState, w_index: int) -> SimState:
        wheel_omega = sim_state.polygon.radius[w_index]

        braking_coeff = 0.1
        braking_d_omega = -jnp.sign(wheel_omega) * braking_coeff
        braking_d_omega = jax.lax.select(
            jnp.abs(braking_d_omega) > jnp.abs(wheel_omega), -wheel_omega, braking_d_omega
        )

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(radius=sim_state.polygon.radius.at[w_index].add(braking_d_omega))
        )

        return sim_state

    def _apply_wheel_impulse(self, sim_state: SimState, w_index: int) -> SimState:
        wheel_omega = sim_state.polygon.radius[w_index]

        w_rot = sim_state.polygon.rotation[w_index]
        w_normal = jnp.array([-jnp.sin(w_rot), jnp.cos(w_rot)])
        ground_speed_along_normal = jnp.dot(w_normal, sim_state.polygon.velocity[w_index])

        wheel_radius = 0.1
        wheel_speed_at_ground = wheel_omega * wheel_radius

        df = wheel_speed_at_ground - ground_speed_along_normal

        # How quick does the wheel catch up to the ground speed
        wheel_omega -= df * 0.6

        # How much does ground speed catch up with the wheel
        friction_wheel_floor = 1.0
        dv = df * friction_wheel_floor * w_normal

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(
                radius=sim_state.polygon.radius.at[w_index].set(wheel_omega),
                velocity=sim_state.polygon.velocity.at[w_index].add(dv),
            )
        )

        return sim_state

    def _apply_wheel_lateral_impulse(self, sim_state: SimState, sim_params: SimParams, w_index: int) -> SimState:
        w_lateral_rot = sim_state.polygon.rotation[w_index] + jnp.pi / 2.0
        w_vel = sim_state.polygon.velocity[w_index]
        w_lateral_normal = jnp.array([-jnp.sin(w_lateral_rot), jnp.cos(w_lateral_rot)])

        sign = jnp.sign(jnp.dot(w_vel, w_lateral_normal))

        # CBA to copy and paste collision code, so we simulate the lateral impulse as a 'collision' with the floor
        collision_manifold = CollisionManifold(
            normal=w_lateral_normal * sign,
            penetration=0.0,
            collision_point=sim_state.polygon.position[w_index],
            active=True,
            acc_impulse_normal=0.0,
            acc_impulse_tangent=0.0,
            restitution_velocity_target=0.0,
        )

        wheel = jax.tree.map(lambda x: x[w_index], sim_state.polygon)
        floor = jax.tree.map(lambda x: x[0], sim_state.polygon)

        w_dv, _, _, _, _, _ = resolve_collision(
            wheel, floor, collision_manifold, does_collide=True, sim_params=sim_params
        )

        slippage_dv_mag = 0.5
        w_dv_mag = jnp.linalg.norm(w_dv)

        w_dv = jax.lax.select(w_dv_mag <= slippage_dv_mag, w_dv, w_dv / w_dv_mag * slippage_dv_mag)

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(velocity=sim_state.polygon.velocity.at[w_index].add(w_dv))
        )

        return sim_state
    
    def _clip_wheel_rotation(self, sim_state: SimState, w_index: int, j_index: int):
        w_rotation = sim_state.polygon.rotation[w_index]
        car_rotation = sim_state.polygon.rotation[self.r_index]

        relative_rotation = w_rotation - car_rotation
        clipped_relative_rotation = jnp.clip(relative_rotation, -jnp.pi / 4, jnp.pi / 4)
        clipped_rotation = car_rotation + clipped_relative_rotation

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(rotation=sim_state.polygon.rotation.at[w_index].set(clipped_rotation)),
            joint=sim_state.joint.replace(rotation=sim_state.joint.rotation.at[j_index].set(clipped_relative_rotation)),
        )

        return sim_state
    
    def create_track(self, rng: chex.PRNGKey, sim_state: SimState, static_sim_params: StaticSimParams) -> SimState:
        
        NUM_CHECKPOINTS = 9    
        
        a = get_random_points_fixed(rng, NUM_CHECKPOINTS, self.max_num_checkpoints)
        # print('a:', a.shape, a)  # between 0 and 1
        x, y, _ = get_bezier_curve_fixed(a, NUM_CHECKPOINTS, self.max_num_checkpoints, self.num_points_per_checkpoint)
        # print('x:', x.shape, x)  # also between 0 and 1
                
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
        print('track:', track.shape, track)
        
        min_x, max_x = jnp.min(x), jnp.max(x)
        min_y, max_y = jnp.min(y), jnp.max(y)
        
        x_offset = (min_x + max_x) / 2.0
        y_offset = (min_y + max_y) / 2.0
        
        def _create_tiles(sim_state: SimState, t1_t2):
            print('t1_t2 shape', t1_t2.shape)
            # t1, t2 = t1_t2
            alpha1, beta1, x1, y1, alpha2, beta2, x2, y2 = t1_t2
            # alpha2, beta2, x2, y2 = t2
            
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
            print('vertices:', vertices)

            # sort verticies clockwise
            center_pos = jnp.mean(vertices, axis=0)
            vertices = vertices - center_pos
            
            angles = jnp.arctan2(p[:, 1], p[:,0]) 
            idxs = jnp.argsort(-angles)  # TODO review
            vertices = vertices.at[idxs].get()

            
            sim_state, (poly_idx, global_idx) = add_polygon_to_scene(
                sim_state,
                static_sim_params,
                center_pos,
                vertices,
                n_vertices=4,
                fixated=True,
            )
            
            return sim_state, (poly_idx, global_idx)
        
        t1_t2 = jnp.column_stack((track[:-1], track[1:]))
        
        sim_state, idxs = jax.lax.scan(_create_tiles, sim_state, t1_t2)
        
        return sim_state

def main():

    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    
    pygame.init()
    screen_surface = pygame.display.set_mode(SCREEN_DIM)

    # Create scene

    renderer = make_render_pixels(static_sim_params, SCREEN_DIM)
    
    car = Car(
        static_sim_params
    )

    rng = jax.random.PRNGKey(0)

    sim_state = car.reset(rng)
    
    action = jnp.zeros((3,))
    while True:

        sim_state = car.step(sim_state, sim_params, action)

        wheel_steer_speed = 0.05
        wheel_steer = 0.0
        if pygame.key.get_pressed()[pygame.K_a]:
            wheel_steer += wheel_steer_speed
        elif pygame.key.get_pressed()[pygame.K_d]:
            wheel_steer -= wheel_steer_speed
        
        d_omega_speed = 0.4
        d_omega = 0.0
        if pygame.key.get_pressed()[pygame.K_w]:
            d_omega = d_omega_speed

        apply_brake = 0
        if pygame.key.get_pressed()[pygame.K_s]:
            apply_brake = 1

        action = jnp.array([wheel_steer, d_omega, apply_brake])
        
        # Render
        pixels = renderer(sim_state)

        # Update screen
        surface = pygame.surfarray.make_surface(np.array(pixels)[:, ::-1])
        screen_surface.blit(surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True


if __name__ == "__main__":
    debug = False

    if debug:
        print("JIT disabled")
        with jax.disable_jit():
            main()
    else:
        main()
