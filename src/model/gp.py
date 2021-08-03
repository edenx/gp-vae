# JAX
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.experimental import stax

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.diagnostics import hpdi

from functools import partial


# kernel functions
# @jit
# def exp_kernel1(x, z, d, var, ls, noise, jitter=1.0e-6):

#     deltaXsq = jnp.power((x[:, None] - z), 2.0)
#     k = var * jnp.exp(-0.5 * deltaXsq / ls)
#     k += (noise + jitter) * jnp.eye(x.shape[0])

#     return k

# check spatial dimension
def check_d(func):
     def reshape(x, z, d, *args, **kwargs):
          # reshape to column vectors
          if d==1:
               # reshape to column vector if d==1
               x = jnp.reshape(x, (x.shape[0], d))
               z = jnp.reshape(z, (z.shape[0], d))
          return func(x, z, d, *args, **kwargs)
     return reshape

# reshape x,z before apply to kernel
@check_d
# @jit
def exp_kernel2(x, z, 
               d,
               var, 
               ls, 
               noise, 
               jitter=1.0e-6,
               include_noise=True
               ):
     # sqaured norm on the spatial dim
     deltaX = jnp.linalg.norm(x[:, None] - z, ord=2, axis=2) 
     k = var * jnp.exp(-0.5 * jnp.power(deltaX/ls, 2.0) )
     # if include_noise:
     if include_noise:
          k += (noise + jitter) * jnp.eye(x.shape[0])
     return k


# approximate k^*
def agg_kernel_grid(rng_key,
                    d, # spatial dim
                    m, # number of MC sample
                    n, # number of intervals on each axis
                    kernel, 
                    var, 
                    ls, 
                    noise,
                    jitter=1.0e-6,
                    ):

    xloc = dist.Uniform()
    grid = jnp.arange(0, 1, 1/n)

    # note that we must have a column vector for spatial locs
    x = xloc.sample(rng_key, (n**d, m, d)) 

    # 1D: sample from [0, 0.1], [0.1, 0.2] etc. uniformly
    if d==1:
      _x = x/n + jnp.expand_dims(grid, axis=(1, 2))

    elif d==2:
      u, v = jnp.meshgrid(grid, grid)
      _x = x/n + jnp.array([[u.flatten()], [v.flatten()]]).transpose((2, 1, 0))
    
    else:
      raise Warning("Function is only implemented for d=1,2")

    _kernel = partial(kernel, d=d, var=var, ls=ls, noise=noise, jitter=jitter)
    __kernel = lambda x, z: jnp.sum(_kernel(x, z))

    # the first dim of sample gives the batch dim, i.e. n**d
    agg__kernel_v1 = vmap(__kernel, (0, None), 0)
    agg__kernel_v2 = vmap(agg__kernel_v1, (None, 0), 1)

    # print(agg__kernel_v1(_x, _x[0]))

    return agg__kernel_v2(_x, _x) / (m ** 2)

# may want to rewrite this to include spatial dim 2
class GP():
     def __init__(
          self, 
          kernel=exp_kernel2, 
          var=1,
          noise=0,
          ls=0.01, # this is default
          jitter=1.0e-6,
          d=1
          ):

          self.kernel = kernel
          self.var = var
          self.noise = noise
          self.ls = ls
          self.jitter = jitter
          self.d = d
     
     def sample(self, ls, x, y=None):
          k = self.kernel(x, x, self.d, self.var, ls, self.noise, self.jitter)

          # sample Y according to the standard gaussian process formula
          numpyro.sample(
               "y",
               dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), 
               obs=y
          )