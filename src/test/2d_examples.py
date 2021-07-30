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

# Modules
from model.gp import GP, exp_kernel2, agg_kernel_grid
from model.vae import VAE
from model.inference import Inference

# draws from GP
def example_gp2(rng_key, n, x, noise=0.001, plot_y=True):
     # we use the default kwargs in GP apart from noise
     gp_y = GP(kernel=exp_kernel2, noise=noise, d=2)
     y = Predictive(gp_y.sample, num_samples=1)(rng_key=rng_key, ls=0.1, x=x)["y"][0]

     ground_truth = None
     # {'x': x, 'y': y}
     
     if plot_y:
          plt.figure()
          plt.imshow(
               y.reshape((n, n)), 
               cmap='viridis',
               interpolation='none', 
               extent=[0,1,0,1],
               origin='lower')
          plt.show()
          plt.close()

          # plt.figure()
          # sns.set_palette('viridis')
          # sns.heatmap(y.reshape((n, n)))
          # plt.show()
          # plt.close()

     return ground_truth, y

    
def main(args, example, dat_noise=0.001):
     """ Plot posterior predictions for varying size of unserved locations
     with index ranging from 1, 10, 100, 200 for a total of 300 (default) 
     dense grid over [0,1].
     args: namespace, args of GP, VAE and Inference
     example: string, example function used to generate data
     dat_noise: float, added noise to generated data"""

     assert args.d is 2

     if args.kernel == "exponential2":
          kernel = exp_kernel2
     else:
          raise NotImplementedError
     
     if example == "gp2":
          example_func = example_gp2
     else:
          raise NotImplementedError

     # n by n grid over [0,1]x[0,1]
     grid = jnp.arange(0, 1, 1/args.n)
     u, v = jnp.meshgrid(grid, grid)
     x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))

     # VAE training
     gp = GP(kernel=kernel, var=args.var, noise=args.noise, d=args.d) # ls is random in training
     vae = VAE(
          gp,
          args.hidden_dims, # list
          args.z_dim, # bottleneck
          args.n**2, # dim==2
          args.batch_size, 
          args.learning_rate,
          args.num_epochs,
          args.num_train,
          args.num_test,
          x,
          args.seed)
     decoder, decoder_params = vae.fit()

     # context points are 1, 10, 100, 200
     obs_idx_dict = {}
     obs_idx_dict['100'] = list(rd.sample(np.arange(args.n**2).tolist(), k=100))
     obs_idx_dict['300'] = (obs_idx_dict['100'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=200)))
     obs_idx_dict['700'] = (obs_idx_dict['300'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=400)))
     obs_idx_dict['1200'] = (obs_idx_dict['700'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=500)))

     rng_key = random.PRNGKey(args.seed+1)

     # generate data for inference
     ground_truth, y = example_func(rng_key, args.n, x, noise=dat_noise, plot_y=False)
     # initialise the Inference object
     inference = Inference(
          decoder, decoder_params,
          args.z_dim, args.d, # n locs on each axis
          x, y, # None for generated data, y_filtered
          obs_idx_dict['1200'],
          args.mcmc_args,
          args.seed,
          ground_truth
          )

     # plot ground truth
     plt.figure(figsize=(15/2, 25/4))
     inference.plot_prediction2(y[None, :])
     plt.title('Ground truth')
     # plt.show()
     plt.savefig('src/test/plots/test_example_gp2_ground_truth.png'.format(example))
     plt.close()

     plt.figure(figsize=(15, 25))

     # Inference
     for i, item in enumerate(obs_idx_dict.items()):
          print(i)
          k, obs_idx = item
          obs_idx = jnp.asarray(obs_idx)

          # mask unobserved y
          mask = jnp.zeros(args.n, dtype=bool).at[obs_idx].set(True)
          unobs_idx = jnp.arange(args.n)[~mask]
          y_filtered = ops.index_update(y, unobs_idx, np.NaN)
          
          # update observation locations
          inference.obs_idx = obs_idx
          inference.y = y_filtered

          # obtain predictions
          prior_pred, pred = inference.fit(plot=False)

          # plot
          ## prior
          plt.subplot(4,2,2*i+1)
          inference.plot_prediction2(prior_pred)
          plt.title('Prior-'+k)

          # posterior
          plt.subplot(4,2,2*i+2)
          inference.plot_prediction2(pred)        
          plt.title('Posterior-'+k)
     
     plt.tight_layout()
     plt.savefig('src/test/plots/test_example_{}.png'.format(example))
     plt.close()


def args_parser():
     parser = argparse.ArgumentParser(description="VAE test")
     # GP
     parser.add_argument("--d", default=2, 
                         type=int, help="GP dimension")
     parser.add_argument("--kernel", default="exponential2", 
                         type=str)
     parser.add_argument("--var", default=1, 
                         type=int, help="marginal variance of kernel")
     parser.add_argument("--ls", default=0.1, 
                         type=float, help="lengthscale of kernel")
     parser.add_argument("--noise", default=0.001, 
                         type=float, help="additional noise of GP")
     # VAE
     parser.add_argument("--n", default=100, 
                         type=int, help="number of point on grid")
     parser.add_argument("--hidden-dims", default=[50,30], 
                         type=list, help="dimension of hidden layers for encoder/decoder")
     parser.add_argument("--z-dim", default=15, 
                         type=int, help="bottleneck dimension")
     parser.add_argument("--batch-size", default=1000, 
                         type=int, help="size of minibatch")
     parser.add_argument("--learning-rate", default=1.0e-3, 
                         type=float)
     parser.add_argument("--num-epochs", default=50, 
                         type=int)
     parser.add_argument("--num-train", default=1000, 
                         type=int)
     parser.add_argument("--num-test", default=1000, 
                         type=int)
     parser.add_argument("--seed", default=0, 
                         type=int, help="seed to generatee rng_keys")
     
     # MCMC
     parser.add_argument("--obs-idx", default=[54, 60, 100],
                         type=list,
                         help="index of observed locations")

     parser.add_argument("--mcmc-args", default={"num_samples": 1000, 
                                                  "num_warmup": 1000,
                                                  "num_chains": 4,
                                                  "thinning": 3}, 
                         type=dict)

     args = parser.parse_args()
     return args


if __name__ == "__main__":

     n = 25 # default in arg parser
     d = 2 
     grid = jnp.arange(0, 1, 1/n)
     rng_key = random.PRNGKey(0)

     if d==1:
          x = grid
          # print(x.shape)
     elif d==2:
          u, v = jnp.meshgrid(grid, grid)
          x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
     else:
          raise NotImplementedError

     example_gp2(rng_key, n, x)

     # args = args_parser()
     # main(args, "gp2")
