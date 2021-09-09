# JAX
import jax.numpy as jnp
from jax import random, jit, ops, vmap
from jax.experimental import stax

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.diagnostics import hpdi

from functools import partial
import numpy as np


# kernel functions
# @jit
# def exp_kernel1(x, z, d, var, ls, noise, jitter=1.0e-6):

#     deltaXsq = jnp.power((x[:, None] - z), 2.0)
#     k = var * jnp.exp(-0.5 * deltaXsq / ls)
#     k += (noise + jitter) * jnp.eye(x.shape[0])

#     return k

# check spatial dimension
def check_d(func):
     """Check spatial dimension of input, transform 1d array to column vector.
     """
     def reshape(x, z, *args, **kwargs):
          assert len(x.shape)==len(z.shape)
          # reshape to column vectors
          if len(x.shape)==1:
               # reshape to column vector if d==1
               x = jnp.reshape(x, (x.shape[0], 1))
               z = jnp.reshape(z, (z.shape[0], 1))
          return func(x, z, *args, **kwargs)
     return reshape

# reshape x,z before apply to kernel
@check_d
# @jit
def exp_kernel2(x, z, 
               # d,
               var, 
               ls, 
               # noise, 
               jitter=1.0e-5
               ):
     """Exponential kernel for 1D and 2D spatial-temporal data.

     Args: 
          x, z (ndarray) - spatial-temporal data.
          d (int) - spatial dimension 1 or 2.
          var (float) - marginal variance.
          ls (float) - lengthscale of the kernel.
          noise (float) - additional noise to diagonal entries.
          jiitte (float) - tiny noise added to diagonal for numerical stability.
          include_noise (bool) - if True include jitter and noise (square Gram matrix).

     Returns:
          kernel gram matrix.
     """
     assert len(x.shape)==len(z.shape)
     assert len(x.shape)==1 or len(x.shape)==2

     # print("exp kernel", x.shape)
     # print(z.shape)

     # sqaured norm on the spatial dim
     deltaX = jnp.linalg.norm(x[:, None] - z, ord=2, axis=2) 
     k = var * jnp.exp(-0.5 * jnp.power(deltaX/ls, 2.0) )

     # ckeck if kernel matrix is a square matrix -- 
     # stablise inversion with jitter on the diagonal
     if k.shape[0] == k.shape[1]:
          k += jitter * jnp.eye(x.shape[0])
     return k


# approximate k^*
def agg_kernel_grid(rng_key,
                    d, # spatial dim
                    m, # number of MC sample
                    n, # number of intervals on each axis
                    kernel, 
                    var, 
                    ls, 
                    # noise,
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

    _kernel = partial(kernel, var=var, ls=ls, jitter=jitter)
    __kernel = lambda x, z: jnp.sum(_kernel(x, z))

    # the first dim of sample gives the batch dim, i.e. n**d
    agg__kernel_v1 = vmap(__kernel, (0, None), 0)
    agg__kernel_v2 = vmap(agg__kernel_v1, (None, 0), 1)

    # print(agg__kernel_v1(_x, _x[0]))

    return agg__kernel_v2(_x, _x) / (m ** 2)

# may want to rewrite this to include spatial dim 2
class GP():
     """Class for GP.

     Attributes:
          kernel - kernel function.
          var (float) - marginal variance of kernel.
          noise (float) - added noise of kernel.
          ls (float) - lengthscale of kernel.
          jitter (float) - small positive noise on diagonal entries.
          d (int) - spatial dimension 1 or 2.
     """
     def __init__(
          self, 
          kernel=exp_kernel2, 
          # var=1,
          # noise=0,
          # ls=0.01, # this is default
          jitter=1.0e-5,
          d=1
          ):

          self.kernel = kernel
          # self.var = var
          # self.noise = noise
          # self.ls = ls
          self.jitter = jitter
          self.d = d
     
     # update the function with user defined variance
     def sample(self, x, y=None, ls=None, var=None, sigma=None):
          """Sample from GP with a given lengthscale and marginal vaiance.

          Args:
               ls (float) - lengthscale of kernel.
               x (ndaray) - spatial location.
               y (ndarray) - (function) value at x.
          
          Returns:
               sampler for y.
          """

          if ls is None:
               ls = numpyro.sample("length", dist.InverseGamma(1,0.1))
          if var is None:
               var = numpyro.sample("var", dist.LogNormal(0.0, 0.1))
          if sigma is None:
               sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

          k = self.kernel(x, x, var, ls, self.jitter)

          ## Sanity check: if length/dx ->1: OK,  if length/dx -> Inf: covariance becomes degenerate
          # logdetK = np.linalg.slogdet(np.asarray(k))[0] * np.linalg.slogdet(np.asarray(k))[1]
          # dx = 1/k.shape[0]
          # print(k[0:5, 0:5])
          # print("dx =" + str(dx))
          # print("log(det(K)) = " + str(logdetK))
          # print("length / dx = " + str(ls/dx))

          # sample Y according to the standard gaussian process formula
          f = numpyro.sample(
               "f",
               dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k)
               )
          numpyro.sample("y", dist.Normal(f, sigma), obs=y)