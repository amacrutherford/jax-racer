import jax
import jax.numpy as jnp


def rmat(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def sv_cross(x, v):
    return jnp.array([-x * v[1], x * v[0]])


def vs_cross(v, x):
    return jnp.array([x * v[1], -x * v[0]])


def zero_to_one(x):
    # We use this to avoid NaNs cropping up in masked out shapes
    return jax.lax.select(x == 0, jnp.ones_like(x), x)

def ccw_only_valid(x: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    large_offset = jnp.where(~valid, 1e3, 0)
    mean_x = jnp.mean(x, axis=0, where=valid[:, None])
    d = x - mean_x
    s = jnp.arctan2(d[:, 1], d[:, 0]) + large_offset
    order = jnp.argsort(s)
    return x.at[order].get()
    
@jax.jit  
def ccw(x: jnp.ndarray) -> jnp.ndarray:
    """ Sort 2d points in counter-clockwise order """
    mean_x = jnp.mean(x, axis=0)
    d = x - mean_x
    s = jnp.arctan2(d[:, 1], d[:, 0])
    order = jnp.argsort(s)
    return x.at[order].get()

@jax.jit
def cw(x: jnp.ndarray) -> jnp.ndarray:
    """ Sort 2d points in clockwise order """
    mean_x = jnp.mean(x, axis=0)
    # print('mean_x:', mean_x)
    d = x - mean_x
    s = jnp.arctan2(d[:, 1], d[:, 0])
    # print('s:', s)
    order = jnp.argsort(-s)
    return x.at[order].get()
    