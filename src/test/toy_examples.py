import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

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


# example dataset for x in [0,1] -- y = x^3 + noise
def example_cubic(rng_key, n, x, noise=0.1, plot_y=True):
     rng_key, rng_key_y = random.split(rng_key)

     # mask = jnp.zeros(n, dtype=bool).at[obs_idx].set(True)
     # unobs_idx = jnp.arange(n)[~mask]

     y = jnp.power(x, 3) + random.normal(rng_key_y, shape=(n, )) * noise
     # y_filtered = ops.index_update(y, unobs_idx, np.NaN)

     ground_truth = {"x":np.arange(0,1,0.001), 
                    "y": np.arange(0,1,0.001)**3}

     # if plot_y:
     #      plt.figure()
     #      plt.plot(ground_truth["x"], 
     #                ground_truth["y"], 
     #                label="y=x^3+noise", c="orange")
     #      plt.scatter(x, y, c="lightgray", label="unobserved")
     #      # plt.scatter(x, y_filtered, color="orange", label="observed")
     #      plt.legend(loc="upper left")  
     #      plt.show()
     #      plt.close()

     return ground_truth, y

# y = sin(x/0.1)
def example_sin(rng_key, n, x, noise=0.1, plot_y=True):

     def func(x):
          return (
               np.sin(x * 3 * 3.14) 
               + 0.3 * np.cos(x * 9 * 3.14) 
               + 0.5 * np.sin(x * 7 * 3.14))

     rng_key, rng_key_y = random.split(rng_key)

     # mask = jnp.zeros(n, dtype=bool).at[obs_idx].set(True)
     # unobs_idx = jnp.arange(n)[~mask]

     y = func(x) + random.normal(rng_key_y, shape=(n, )) * noise
     # y_filtered = ops.index_update(y, unobs_idx, np.NaN)

     ground_truth = {"x":np.arange(0,1,0.001), 
                    "y": func(np.arange(0,1,0.001))}
     # if plot_y:
     #      plt.figure()
     #      plt.plot(ground_truth["x"], 
     #                ground_truth["y"], c="orange")
     #      plt.scatter(x, y, c="lightgray", label="unobserved")
     #      # plt.scatter(x, y_filtered, color="orange", label="observed")
     #      plt.legend(loc="upper left")  
     #      plt.show()
     #      plt.close()

     return ground_truth, y

def example_gp(rng_key, n, x, plot_y=True):
     # we use the default argument in GP
     # obs_idx = jnp.asarray(obs_idx)
     # mask = jnp.zeros(n, dtype=bool).at[obs_idx].set(True)
     # unobs_idx = jnp.arange(n)[~mask]

     gp_y = GP(kernel=exp_kernel1)
     y = Predictive(gp_y.sample, num_samples=1)(rng_key=rng_key, ls=gp_y.ls, x=x)["y"][0]
     # y_filtered = ops.index_update(y, unobs_idx, np.NaN)

     # 1000 represents number of points to make between x.min and x.max
     x_new = np.linspace(x.min(), x.max(), 40) 
     spl = make_interp_spline(x, y, k=3)  # type: BSpline
     power_smooth = spl(x_new)
     ground_truth = {"x":x_new, 
                    "y": power_smooth}

     # if plot_y:
     #      plt.figure()
     #      plt.plot(ground_truth["x"], 
     #                ground_truth["y"], 
     #                label="y=gp+noise")
     #      plt.scatter(x, y, color="lightgray", label="unobserved")
     #      # plt.scatter(x, y_filtered, color="orange", label="observed")
     #      plt.legend(loc="upper left")  
     #      plt.show()
     #      plt.close()

     return ground_truth, y
     
def main(args, example_func):
     assert args.d is 1
     # train VAE with n points
     x = jnp.arange(0, 1, 1/args.n)

     if args.kernel is "exponential":
          kernel = exp_kernel1
     else:
          raise Warning("Othr kernels are not implemented")
     
     # VAE training
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
     decoder, decoder_params = vae.fit()

     # context points are 1, 10, 100 
     obs_idx_dict = {}
     obs_idx_dict['1'] = [100]
     obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
     obs_idx_dict['100'] = list(rd.sample(np.arange(300).tolist(), k=100))
     obs_idx_dict['200'] = list(rd.sample(np.arange(300).tolist(), k=200))

     rng_key = random.PRNGKey(args.seed+1)

     # generate data for inference
     ground_truth, y = example_func(rng_key, args.n, x)

     # initialise the Inference object
     inference = Inference(
          decoder, decoder_params,
          args.z_dim,
          x, None,
          args.obs_idx,
          args.mcmc_args,
          args.seed,
          ground_truth
          )

     plt.figure(figsize=(15, 20))


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
          # plt.subplot(3,2,2*i+1)
          # inference.plot_prediction(prior_pred)
          # plt.title('Prior-'+k)

          # plt.subplot(3,2,2*i+2)
          plt.subplot(2,2,i+1)
          inference.plot_prediction(pred)
          plt.title('Posterior-'+k)
     
     plt.tight_layout()
     # plt.show()
     plt.savefig('plots/test_example_sin.png')
     plt.close()


def args_parser():
     parser = argparse.ArgumentParser(description="VAE test")
     # GP
     parser.add_argument("--d", default=1, 
                         type=int, help="GP dimension")
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
     parser.add_argument("--obs-idx", default=[54, 60, 100],
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

     args = args_parser()
     main(args, example_sin)