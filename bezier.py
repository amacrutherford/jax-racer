""" 
Drawn from: https://github.com/facebookresearch/dcd/blob/main/envs/box2d/bezier.py
"""

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from functools import partial
from flax import struct
import jax
import jax.numpy as jnp

bernstein = lambda n, k, t: binom(n, k) * t**k * (1.0 - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1), self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi), self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a=None, rad=0.2, edgy=0, **kw):
    """Given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    if a is None:
        a = get_random_points(**kw)

    numpoints = kw.get("numpoints", 30)

    p = np.arctan(edgy) / np.pi + 0.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var", numpoints=numpoints)
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0, **kw):
    """Create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7 / n
    np_random = kw.get("np_random", np.random)
    a = np_random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1, np_random=np_random)


@partial(jax.jit, static_argnames=("max_n", "num_to_generate"))
def get_random_points_fixed(rng, n: int, max_n: int, mindst: float = 0.1, scale=0.8, num_to_generate=10):
    idxs = jnp.arange(max_n)
    valid = idxs < n
    large_offset = jnp.where(idxs >= n, 1e3, 0)

    def _gen_points(rng):
        a = jax.random.uniform(rng, shape=(max_n, 2))
        # sort points in ccw order
        mean_a = jnp.mean(a, axis=0, where=valid[:, None])
        d = a - mean_a
        s = jnp.arctan2(d[:, 1], d[:, 0]) + large_offset
        order = jnp.argsort(s)
        a = a.at[order].get()

        a = jnp.where(valid[:, None], a, a.at[0].get())

        diff = jnp.diff(a, axis=0)

        d = jnp.sqrt(jnp.sum(diff**2, axis=1))
        points_valid = (d >= mindst) + ~valid[:-1]
        return a, jnp.all(points_valid)

    rngs = jax.random.split(rng, num_to_generate)
    a, a_valid = jax.vmap(_gen_points)(rngs)
    selected_a = jnp.argmax(a_valid)
    return a.at[selected_a].get()

@partial(jax.jit, static_argnames=("max_n", "numpoints"))
def get_bezier_curve_fixed(a: jnp.ndarray, n: int, max_n: int, numpoints: int = 30, rad=0.2, edgy=0):
    """Given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    # if a is None:
    #     a = get_random_points_fixed(**kw)

    idxs = jnp.arange(max_n)
    # valid = idxs < n

    p = jnp.arctan(edgy) / jnp.pi + 0.5
    # a = ccw_sort(a)
    a = jnp.append(a, jnp.atleast_2d(a[0, :]), axis=0)
    d = jnp.diff(a, axis=0)
    ang = jnp.arctan2(d[:, 1], d[:, 0])
    ang = jnp.where(idxs < n, ang, ang.at[0].get())
    # print("ang", ang)

    # f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    # def _scale_angle(ang):
    #     return jnp.where(ang >= 0, ang, ang + 2 * np.pi)
    ang = jnp.where(ang >= 0, ang, ang + 2 * jnp.pi)
    ang1 = ang
    # print("ang1", ang1)
    ang2 = jnp.roll(ang, 1)
    # print("ang2", ang2)
    ang = p * ang1 + (1 - p) * ang2 + (jnp.abs(ang2 - ang1) > jnp.pi) * jnp.pi
    ang = jnp.append(ang, ang[0])
    # print("ang", ang)

    a = jnp.append(a, jnp.atleast_2d(ang).T, axis=1)

    seg_gen = JaxSegment(numpoints=numpoints, r=rad)

    segments = jax.vmap(seg_gen.calc_curve)(a[:-1, :2], a[1:, :2], a[:-1, 2], a[1:, 2])
    curve = segments.reshape((-1, 2))

    # TODO mask out invalid points

    x, y = curve.T
    return x, y, a


class JaxSegment:
    def __init__(self, numpoints=30, r=0.3):

        self.numpoints = numpoints
        self.r = r

    partial(jax.jit, static_argnums=(0,))

    def calc_curve(self, p1, p2, angle1, angle2):
        d = jnp.sqrt(jnp.sum((p2 - p1) ** 2))
        r = self.r * d

        points = jnp.zeros((4, 2))
        points = points.at[0].set(p1)
        points = points.at[3].set(p2)
        points = points.at[1].set(p1 + jnp.array([r * jnp.cos(angle1), r * jnp.sin(angle1)]))
        points = points.at[2].set(p2 + jnp.array([r * jnp.cos(angle2 + jnp.pi), r * jnp.sin(angle2 + jnp.pi)]))

        bernstein = lambda n, k, t: binom(n, k) * t**k * (1.0 - t) ** (n - k)

        N = len(points)
        t = np.linspace(0, 1, num=self.numpoints)
        curve = np.zeros((self.numpoints, 2))

        def _body(curve, i):
            return curve + jnp.outer(bernstein(N - 1, i, t), points[i]), None

        curve, _ = jax.lax.scan(_body, curve, jnp.arange(N))

        return curve


from jax.scipy.special import gammaln


def binom(x, y):
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    MAX_NUM_CHECKPOINTS = 8
    NUM_CHECKPOINTS = 7
    NUM_POINTS = 30

    rad = 0.2
    edgy = 0.5

    for c in np.array([[0, 0], [0, 1]]):
        a = get_random_points(n=NUM_CHECKPOINTS, scale=0.5) + c
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        plt.plot(x, y)

    rng = jax.random.PRNGKey(1)
    for c in np.array([[1, 0], [1, 1]]):
        rng, _rng = jax.random.split(rng)
        a = get_random_points_fixed(_rng, NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS, 0.1) + c
        print("a", a)
        
        x, y, _ = get_bezier_curve_fixed(a, NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS, NUM_POINTS)
        print("x", x)
        print("y", y.shape)
        
        x = x[:NUM_CHECKPOINTS*NUM_POINTS]
        y = y[:NUM_CHECKPOINTS*NUM_POINTS]
        
        # raise
        plt.plot(x, y)

    plt.savefig("bezier.png")
