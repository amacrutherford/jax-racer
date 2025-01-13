import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_circle,
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

from jax2d.collision import resolve_collision
from jax2d.engine import PhysicsEngine, create_empty_sim
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


def main():
    screen_dim = (500, 500)

    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)
    sim_state = sim_state.replace(gravity=jnp.zeros(2))

    # Create a rectangle for the car body
    car_dims = jnp.array([0.15, 0.3])
    car_half_dims = car_dims / 2.0
    car_pos = jnp.array([2.5, 2.5])

    sim_state, (_, r_index) = add_rectangle_to_scene(
        sim_state, static_sim_params, position=car_pos, dimensions=car_dims
    )

    # Create the wheels
    wheel_dims = jnp.array([0.04, 0.08])

    sim_state, (_, w0_index) = add_rectangle_to_scene(
        sim_state,
        static_sim_params,
        position=car_pos + car_half_dims * jnp.array([1.0, 1.0]),
        dimensions=wheel_dims,
        friction=0.0,
    )

    sim_state, (_, w1_index) = add_rectangle_to_scene(
        sim_state,
        static_sim_params,
        position=car_pos + car_half_dims * jnp.array([1.0, -1.0]),
        dimensions=wheel_dims,
        friction=0.0,
    )

    sim_state, (_, w2_index) = add_rectangle_to_scene(
        sim_state,
        static_sim_params,
        position=car_pos + car_half_dims * jnp.array([-1.0, -1.0]),
        dimensions=wheel_dims,
        friction=0.0,
    )

    sim_state, (_, w3_index) = add_rectangle_to_scene(
        sim_state,
        static_sim_params,
        position=car_pos + car_half_dims * jnp.array([-1.0, 1.0]),
        dimensions=wheel_dims,
        friction=0.0,
    )

    # # Join the wheels to the car body with revolute joints
    # # Relative positions are from the centre of masses of each object
    sim_state, _ = add_fixed_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=w0_index,
        a_relative_pos=car_half_dims * jnp.array([1.0, 1.0]),
        b_relative_pos=jnp.zeros(2),
    )

    sim_state, _ = add_fixed_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=w1_index,
        a_relative_pos=car_half_dims * jnp.array([1.0, -1.0]),
        b_relative_pos=jnp.zeros(2),
    )

    sim_state, _ = add_fixed_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=w2_index,
        a_relative_pos=car_half_dims * jnp.array([-1.0, -1.0]),
        b_relative_pos=jnp.zeros(2),
    )

    sim_state, _ = add_fixed_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=w3_index,
        a_relative_pos=car_half_dims * jnp.array([-1.0, 1.0]),
        b_relative_pos=jnp.zeros(2),
    )

    # Renderer
    renderer = make_render_pixels(static_sim_params, screen_dim)

    # Step scene
    step_fn = jax.jit(engine.step)

    # Car racing fns
    @jax.jit
    def _apply_wheel_impulse(sim_state, w_index, dir):
        max_forward_speed = 3.0
        max_backward_speed = max_forward_speed / 3.0
        drive_speed = 1.0

        w_rot = sim_state.polygon.rotation[w_index]
        w_normal = jnp.array([-jnp.sin(w_rot), jnp.cos(w_rot)])
        w_dv = w_normal * dir * drive_speed

        current_v = sim_state.polygon.velocity[w_index]
        w_speed_along_normal = jnp.dot(current_v, w_normal)
        new_velocity = current_v + w_dv

        can_apply = ((dir > 0) & (w_speed_along_normal < max_forward_speed)) | (
            (dir < 0) & (w_speed_along_normal > -max_backward_speed)
        )

        new_velocity = jax.lax.select(
            can_apply,
            new_velocity,
            current_v,
        )

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(velocity=sim_state.polygon.velocity.at[w_index].set(new_velocity))
        )

        return sim_state

    @jax.jit
    def apply_wheel_lateral_impulse(sim_state, w_index):
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

        # print(w_dv)

        return sim_state

    @jax.jit
    def _clip_wheel_rotation(sim_state, w_index):
        w_rotation = sim_state.polygon.rotation[w_index]
        car_rotation = sim_state.polygon.rotation[r_index]

        relative_rotation = w_rotation - car_rotation
        clipped_relative_rotation = jnp.clip(relative_rotation, -jnp.pi / 4, jnp.pi / 4)
        clipped_rotation = car_rotation + clipped_relative_rotation
        # clipped_rotation = w_rotation

        sim_state = sim_state.replace(
            polygon=sim_state.polygon.replace(rotation=sim_state.polygon.rotation.at[w_index].set(clipped_rotation))
        )

        return sim_state

    pygame.init()
    screen_surface = pygame.display.set_mode(screen_dim)

    while True:
        actions = -jnp.ones(static_sim_params.num_joints + static_sim_params.num_thrusters)
        sim_state, _ = step_fn(sim_state, sim_params, actions)

        # Take action
        wheel_rotate_speed = 0.05
        if pygame.key.get_pressed()[pygame.K_a]:
            sim_state = sim_state.replace(
                polygon=sim_state.polygon.replace(
                    rotation=sim_state.polygon.rotation.at[w0_index].add(wheel_rotate_speed)
                )
            )
            sim_state = sim_state.replace(
                polygon=sim_state.polygon.replace(
                    rotation=sim_state.polygon.rotation.at[w3_index].add(wheel_rotate_speed)
                )
            )

        if pygame.key.get_pressed()[pygame.K_d]:
            sim_state = sim_state.replace(
                polygon=sim_state.polygon.replace(
                    rotation=sim_state.polygon.rotation.at[w0_index].add(-wheel_rotate_speed)
                )
            )
            sim_state = sim_state.replace(
                polygon=sim_state.polygon.replace(
                    rotation=sim_state.polygon.rotation.at[w3_index].add(-wheel_rotate_speed)
                )
            )

        # Cap wheel rotations
        sim_state = _clip_wheel_rotation(sim_state, w0_index)
        sim_state = _clip_wheel_rotation(sim_state, w3_index)

        if pygame.key.get_pressed()[pygame.K_w]:
            sim_state = _apply_wheel_impulse(sim_state, w1_index, 1.0)
            sim_state = _apply_wheel_impulse(sim_state, w2_index, 1.0)

        if pygame.key.get_pressed()[pygame.K_s]:
            sim_state = _apply_wheel_impulse(sim_state, w1_index, -1.0)
            sim_state = _apply_wheel_impulse(sim_state, w2_index, -1.0)

        # Cancel out lateral movement
        sim_state = apply_wheel_lateral_impulse(sim_state, w0_index)
        sim_state = apply_wheel_lateral_impulse(sim_state, w1_index)
        sim_state = apply_wheel_lateral_impulse(sim_state, w2_index)
        sim_state = apply_wheel_lateral_impulse(sim_state, w3_index)

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
