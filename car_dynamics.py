"""
Drawn from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_dynamics.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import pygame
from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_circle,
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

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
from jax2d.sim_state import StaticSimParams, SimParams, SimState


FORWARD = jnp.array([0, 1])
SIDE = jnp.array([1, 0])

ENGINE_POWER = 4e4
WHEEL_MOMENT_OF_INERTIA = 1.6
FRICTION_LIMIT = 400

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

    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

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

        rect_colours = jnp.ones((static_sim_params.num_polygons, 3)) * 128.0
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
class CarState:
    gas: float
    brake: float
    steer: float
    sim_state: SimState
    w_phase: float = 0.0
    w_omega: float = 0.0

class Car:
    def __init__(self,
                 width: float,
                 radius: float,
                 static_sim_params: StaticSimParams):
        self.static_sim_params = static_sim_params
        self.width = width
        self.radius = radius
        self.wheel_poly = jnp.array(
            [[-self.width / 2, -self.radius], [self.width / 2, -self.radius], [self.width / 2, self.radius], [-self.width / 2, self.radius]]
        )
        self.dt = 0.01

    def add_wheel_to_scene(self, sim_state, static_sim_params, position, angle):

        sim_state, (polygon_index, global_index) = add_polygon_to_scene(sim_state, static_sim_params, position, self.wheel_poly, n_vertices=4, rotation=angle)
        print('poly and global index', polygon_index, global_index)
        return sim_state, (polygon_index, global_index)
    
    def reset(self):

        sim_state = create_empty_car_sim(self.static_sim_params, add_floor=False, add_walls_and_ceiling=False)

        sim_state, (_, _) = self.add_wheel_to_scene(sim_state, self.static_sim_params, jnp.array([1.5, 1.0]), 0.0)

        car_state = CarState(
            gas=0.0,
            brake=0.0,
            steer=0.0,
            sim_state=sim_state
        )

        return car_state

    def step(self, car_state: CarState):

        gas_act = 0.2  # (0, 1)
        gas_act = jnp.clip(gas_act, 0.0, 1.0)

        brake_act = 0.0  # (0, 1)
        
        steer_act = 0.1  # (-1, 1)

        # TODO below to be amend across all wheels
        
        # TODO get friction limit from position
        
        
        diff = gas_act - car_state.gas
        gas = car_state.gas + jax.lax.select(diff > 0.1, 0.1, diff)  # gradually increase

        brake = brake_act

        # TODO add steering logic
        
        print('polygon rotation', car_state.sim_state.polygon.rotation)

        # get forward and side vectors in wheel frame
        forw = rmat(car_state.sim_state.polygon.rotation[0]) @ FORWARD
        side = rmat(car_state.sim_state.polygon.rotation[0]) @ SIDE

        v = car_state.sim_state.polygon.velocity[0]
        w = car_state.sim_state.polygon.angular_velocity[0]
        
        # get forward and side velocity components
        vf = jnp.dot(v, forw)
        vs = jnp.dot(v, side) 
        
        print('vf, vs', vf, vs)
                
        # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
        # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
        # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
        # add small coef not to divide by zero
        w_omega = car_state.w_omega + self.dt * ENGINE_POWER * gas / WHEEL_MOMENT_OF_INERTIA / (jnp.abs(car_state.w_omega) + 5.0)
        
        # brake logic
        def _brake(w_omega, brake):
            BRAKE_FORCE = 15  # rad/s
            dir = -jnp.sign(w_omega)
            val = BRAKE_FORCE * brake
            val = jax.lax.select(jnp.abs(val) > jnp.abs(w_omega), jnp.abs(w_omega), val)
            return w_omega + dir * val
        
        w_omega = jax.lax.select(brake > 0.0, _brake(w_omega, brake), w_omega)
        w_omega = jax.lax.select(brake >= 0.9, 0.0, w_omega)  # straight to 0
        
        w_phase = car_state.w_phase + self.dt * w_omega  
         
        vr = w_omega * self.radius  # velocity of the wheel at the contact point
        f_force = -vf + vr  # difference in intended and actual velocity, negative means brake
        p_force = -vs  # side force, cancel out lateral velocity up to a point (friction limit)
        
        # Physically correct is to always apply friction_limit until speed is equal.
        # But dt is finite, that will lead to oscillations if difference is already near zero.
        # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)    
        f_force *= 82
        p_force *= 82
        force = jnp.sqrt(f_force ** 2 + p_force ** 2)
        
        # TODO add skid trace logic https://github.com/Farama-Foundation/Gymnasium/blob/c6c581538f84091238192d5db555cd661efcee86/gymnasium/envs/box2d/car_dynamics.py#L232
        
        def _over_limit_force(total_force: float, dir_force: float) -> float:
            dir_force /= total_force
            dir_force *= FRICTION_LIMIT
            return dir_force
        
        f_force = jax.lax.select(force > FRICTION_LIMIT, _over_limit_force(force, f_force), f_force)
        p_force = jax.lax.select(force > FRICTION_LIMIT, _over_limit_force(force, p_force), p_force)
        
        w_omega -= self.dt * f_force * self.radius / WHEEL_MOMENT_OF_INERTIA
        
        # apply forces
        # assume this is add a thruster to the centre of the wheel and apply force via that or a motor on the joint? 

def main():
    screen_dim = (500, 500)

    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)

    car = Car(
        0.1, 0.3, static_sim_params
    )

    car_state = car.reset()
    
    car.step(car_state)

    raise

    # Create circles for the wheels of the car
    sim_state, (_, c1_index) = add_circle_to_scene(
        sim_state, static_sim_params, position=jnp.array([1.5, 1.0]), radius=0.35
    )
    sim_state, (_, c2_index) = add_circle_to_scene(
        sim_state, static_sim_params, position=jnp.array([2.5, 1.0]), radius=0.35
    )

    # Join the wheels to the car body with revolute joints
    # Relative positions are from the centre of masses of each object
    sim_state, _ = add_revolute_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=c1_index,
        a_relative_pos=jnp.array([-0.5, 0.0]),
        b_relative_pos=jnp.zeros(2),
        motor_on=True,
    )
    sim_state, _ = add_revolute_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=c2_index,
        a_relative_pos=jnp.array([0.5, 0.0]),
        b_relative_pos=jnp.zeros(2),
        motor_on=True,
    )

    # Add a triangle for a ramp - we fixate the ramp so it can't move
    triangle_vertices = jnp.array(
        [
            [0.5, 0.1],
            [0.5, -0.1],
            [-0.5, -0.1],
        ]
    )
    sim_state, (_, t1) = add_polygon_to_scene(
        sim_state,
        static_sim_params,
        position=jnp.array([2.7, 0.1]),
        vertices=triangle_vertices,
        n_vertices=3,
        fixated=True,
    )

    # Renderer
    renderer = make_render_pixels(static_sim_params, screen_dim)

    # Step scene
    step_fn = jax.jit(engine.step)

    pygame.init()
    screen_surface = pygame.display.set_mode(screen_dim)

    while True:
        actions = -jnp.ones(static_sim_params.num_joints + static_sim_params.num_thrusters)
        sim_state, _ = step_fn(sim_state, sim_params, actions)

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
