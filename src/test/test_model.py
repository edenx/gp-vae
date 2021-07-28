import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# general libraries
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from scipy.interpolate import make_interp_spline, BSpline

# JAX
import jax
from jax import random, ops
import jax.numpy as jnp

# Numpyro
from numpyro.infer import Predictive
import numpyro.distributions as dist

from model.gp import GP, exp_kernel1, agg_kernel_grid
from model.vae import VAE
from model.inference import Inference

from toy_examples import example_cubic, example_sin, example_gp

# Test GP ----------------------------------------------------------------
def test_GP(
     d, n,
     rng_key,
     kernel, 
     kernel_args,
     num_samples,
     random_ls
     ):
     assert d==1 or d==2

     grid = jnp.arange(0, 1, 1/n)
     if d==1:
          x = grid
     elif d==2:
          u, v = jnp.meshgrid(grid, grid)
          x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))

     gp = GP(kernel=kernel)
     gp_predictive = Predictive(gp.sample, num_samples=num_samples)
          
     samples = gp_predictive(rng_key=rng_key, x=x)

     if d==1:
          plt.figure()
          for i in range(num_samples):
               plt.plot(x, samples['y'][i])
          plt.show()
          plt.close()
     elif d==2:
          for i in range(num_samples):
               plt.figure()
               sns.heatmap(samples['y'][i].reshape((n, n)))
               plt.show()
               plt.close()


# Test VAE ----------------------------------------------------------------------------
def test_VAE(x, args):

     assert args.d==1

     if args.kernel is "exponential":
          kernel = exp_kernel1
     else:
          raise Warning("Othr kernels are not implemented")

     gp = GP(kernel=kernel, var=args.var, noise=args.noise)
     vae = VAE(
          gp,
          args.hidden_dims, # list
          args.z_dim, # bottleneck
          args.n,
          args.batch_size, 
          args.learning_rate,
          args.num_epochs,
          args.num_train,
          args.num_test,
          x,
          args.seed)

     return vae.fit()
     
# test Inference -------------------------------------------------------------------
def test_Inference(example_func, decoder, decoder_params, x, args):
     
     assert args.d==1 
     rng_key = random.PRNGKey(1)
     ground_truth, y = example_func(rng_key, args.n, x, args.obs_idx)

     # inference on unobserved y with trained decoder
     inference = Inference(
          decoder, decoder_params,
          args.z_dim,
          x, y,
          args.obs_idx,
          args.mcmc_args,
          args.seed,
          ground_truth
     )

     inference.fit()


def args_parser():
     parser = argparse.ArgumentParser(description="VAE test")
     # GP
     parser.add_argument("--d", default=1, 
                         type=int, help="GP dimension")
     # parser.add_argument("--random-ls", default=True, 
     #                     type=bool, help="If to randomly generate lengthscale")
     parser.add_argument("--kernel", default="exponential", 
                         type=str)
     parser.add_argument("--var", default=1, 
                         type=int, help="marginal variance of kernel")
     parser.add_argument("--ls", default=0.001, 
                         type=float, help="lengthscale of kernel")
     parser.add_argument("--noise", default=0.001, 
                         type=float, help="additional noise of GP")
     # VAE
     parser.add_argument("--n", default=300, 
                         type=int, help="number of point on grid")
     parser.add_argument("--hidden-dims", default=[35,30], 
                         type=list, help="dimension of hidden layers for encoder/decoder")
     parser.add_argument("--z-dim", default=10, 
                         type=int, help="bottleneck dimension")
     parser.add_argument("--batch-size", default=1000, 
                         type=int, help="size of minibatch")
     parser.add_argument("--learning-rate", default=1.0e-3, 
                         type=float)
     parser.add_argument("--num-epochs", default=20, 
                         type=int)
     parser.add_argument("--num-train", default=1000, 
                         type=int)
     parser.add_argument("--num-test", default=1000, 
                         type=int)
     parser.add_argument("--seed", default=0, 
                         type=int, help="seed to generatee rng_keys")
     
     # MCMC
     parser.add_argument("--obs-idx", default=
     [54, 60, 100],
     # [0, 20, 23, 30, 35, 50, 54, 100, 108, 110, 117, 125, 130, 133, 140, 143, 146, 170, 190, 195, 200, 207, 240, 250], 
                         type=list,
                         help="dimension of hidden layers for encoder/decoder")

     parser.add_argument("--mcmc-args", default={"num_samples": 1000, 
                                                  "num_warmup": 1000,
                                                  "num_chains": 4,
                                                  "thinning": 3}, 
                         type=dict)

     args = parser.parse_args()
     return args


if __name__ == "__main__":

     # test class GP --------------------------------------------------------------------
     # d = 1
     # n = 100 # number of points on each axis
     # kernel_args = {"var": 1, "ls": 0.001, "noise": 0.001}
     # # the added noise should be of scale ~0.001 in order to get stabiliity when sampling
     # num_samples = 5
     # rng_key = random.PRNGKey(19)

     # test_GP(
     #      d, n,
     #      rng_key,
     #      exp_kernel1, 
     #      kernel_args,
     #      num_samples=num_samples,
     #      random_ls=True
     #      )

     # test class VAE ------------------------------------------------------------------
     # generate x's
     n = 300 # default in arg parser
     d = 1 # default spatial dim
     grid = jnp.arange(0, 1, 1/n)
     if d==1:
          x = grid
          # print(x.shape)
     elif d==2:
          u, v = jnp.meshgrid(grid, grid)
          x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
     
     args = args_parser()
     # ground_truth, y = example_gp(random.PRNGKey(10), args.n, x, args.obs_idx)

     decoder_nn, decoder_params = test_VAE(x, args)

     # test class Inference ------------------------------------------------------------
     test_Inference(example_gp, decoder_nn, decoder_params, x, args)
