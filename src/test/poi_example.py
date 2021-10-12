import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# general libraries
import time
import numpy as np
import pickle
from functools import partial
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from math import ceil

# JAX
import jax
from jax import random, ops
import jax.numpy as jnp

# Numpyro
from numpyro.infer import Predictive
import numpyro.distributions as dist

# Modules
from model.gp import PoiGP, exp_kernel2, agg_kernel_grid
from model.vae import PoiVAE
from model.inference import Inference, PoiVAEInference

def plot_samps(x, y, rate, num_samp, columns=4, ttl="Examples of data we want to encode"):
     if num_samp > 1:
          rows = ceil(num_samp / columns)

          fig, axs = plt.subplots(rows, columns, figsize=(19, 4*rows))

          for r in range(rows):
               for c in range(columns):
                    axs[r, c].scatter(x, y[r*columns + c].T, color="green", label="Poisson data")
                    axs[r, c].plot(x, rate[r*columns + c].T, color="orange", label="rate=exp(GP)")
                    axs[r, c].legend(loc = 'upper right')
                    for i in range(np.max(y[r*columns + c])):
                         axs[r, c].axhline(i, color="grey", linestyle='dashed', linewidth=0.8)
          fig.suptitle(ttl, fontsize=16)
          fig.show()
          fig.close()
     else:
          plt.figure()
          plt.scatter(x, y, color="orange", label="Poisson data")
          plt.plot(x, rate, label="rate=exp(GP)")
          plt.legend(loc = 'upper right')
          plt.show()
          plt.close()


# draws from GP
def example_poigp(
     rng_key, 
     x, 
     ls=0.05, # ls=0.07,
     var=None, # var=0.5,
     noise=None, # sigma=0, 
     jitter=1.0e-5,
     plot_y=True,
     num_samp=1
     ):
     # we use the default kwargs in GP apart from noise
     gp_y = PoiGP(kernel=exp_kernel2, jitter=jitter, d=1)
     predictive = Predictive(gp_y.sample, num_samples=num_samp)(rng_key=rng_key, x=x, ls=ls, var=var, sigma=noise)
     
     y = predictive["y"]
     rate = predictive["rate"]
     ground_truth = {"x": x, 
                    "y": y[0],
                    "rate": rate}
     
     if plot_y:
          plot_samps(x, y, rate, num_samp)

     return ground_truth, y[0]


def poivae_mcmc(args, x, example, dat_noise=None):
     assert args.d is 1

     if example == "poigp":
          example_func = example_poigp
     else:
          raise NotImplementedError

     rng_key = random.PRNGKey(args.seed+1)
     # generate data for inference
     ground_truth, y = example_func(rng_key, x, noise=dat_noise, plot_y=False)

     # setup GP ---------------------------------------------------------------
     if args.kernel == "exponential2":
          kernel = exp_kernel2
     else:
          raise NotImplementedError

     # ls, var and noise are random in training
     gp = PoiGP(
          kernel=kernel, 
          jitter=args.jitter, 
          d=args.d)

     # VAE training -----------------------------------------------------------
     vae = PoiVAE(
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

     decoder = vae.vae_decoder()[1]
     decoder_params = vae.fit(plot_loss=True)
     # Save decoder params
     with open('src/test/decoders/decoder_parmas_1d_n300_poi', 'wb') as file:
          pickle.dump(decoder_params, file)

      # # open decoder params from file
     # with open('src/test/decoders/decoder_parmas_1d_n300_poi', 'rb') as file:
     #      decoder_params = pickle.load(file)

     # Inference --------------------------------------------------------------

     # initialise the Inference object
     inference = PoiVAEInference(
          decoder, decoder_params,
          args.z_dim, args.d,
          x, None, # None for generated data, y_filtered
          args.obs_idx,
          args.mcmc_args,
          args.seed,
          ground_truth
          )

     # plot samples fromtrained VAE prior -------------------------------------
     vae_predictive = Predictive(inference.regression, num_samples=10)
     rng_key, rng_key_predict = random.split(random.PRNGKey(1))
     vae_draws = vae_predictive(rng_key_predict)
     y_draws = vae_draws['y_pred']
     rate_draws = vae_draws['rate']

     plt.figure()
     for i in range(10):
          plot_samps(x, y_draws, rate_draws)
     plt.show()
     plt.close()

     # posterior plot for increasing number of context pnts --------------------
     # context points are 1, 10, 40, 100
     obs_idx_dict = {}
     obs_idx_dict['1'] = [100]
     obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
     obs_idx_dict['40'] = list(rd.sample(np.arange(args.n).tolist(), k=40))
     obs_idx_dict['100'] = list(rd.sample(np.arange(args.n).tolist(), k=100))

     plt.figure(figsize=(15, 12.5))
     for i, item in enumerate(obs_idx_dict.items()):
          print(i)
          k, obs_idx = item
 
          obs_idx = jnp.asarray(obs_idx)
          # mask unobserved y
          mask = jnp.zeros(args.n, dtype=bool).at[obs_idx].set(True)
          unobs_idx = jnp.arange(args.n)[~mask]
          y_filtered = ops.index_update(y, ops.index[unobs_idx], np.nan)

     
          # update observation locations
          inference.obs_idx = obs_idx
          inference.y = y_filtered

          # obtain predictions
          prior_pred, pred = inference.fit(plot=False)


          ## posterior
          plt.subplot(2,2,i+1)
          inference.plot_prediction(pred)
          plt.title('Posterior-'+k)
     
     plt.tight_layout()
     plt.savefig('src/test/plots/vae_mcmc_{}.png'.format(example))
     plt.show()
     plt.close()




def args_parser():
     parser = argparse.ArgumentParser(description="VAE test")
     # GP
     parser.add_argument("--d", default=1, 
                         type=int, help="GP dimension")
     parser.add_argument("--kernel", default="exponential2", 
                         type=str)
     parser.add_argument("--var", default=1, 
                         type=int, help="marginal variance of kernel")
     parser.add_argument("--ls", default=0.27, 
                         type=float, help="lengthscale of kernel")
     parser.add_argument("--sigma", default=0.0, 
                         type=float, help="random noise of training sample")
     parser.add_argument("--jitter", default=1.0e-4, 
                         type=float, help="fixed additional noise to kernel diagonal for numerical stability")
     # VAE
     parser.add_argument("--n", default=300, 
                         type=int, help="number of point on grid")
     parser.add_argument("--hidden-dims", default=[35,30], 
                         type=list, help="dimension of hidden layers for encoder/decoder")
     parser.add_argument("--z-dim", default=10, 
                         type=int, help="bottleneck dimension")
     parser.add_argument("--batch-size", default=100, 
                         type=int, help="size of minibatch")
     parser.add_argument("--learning-rate", default=1.0e-3, 
                         type=float)
     parser.add_argument("--num-epochs", default=70, 
                         type=int)
     parser.add_argument("--num-train", default=1000, 
                         type=int)
     parser.add_argument("--num-test", default=1000, 
                         type=int)
     parser.add_argument("--seed", default=1, 
                         type=int, help="seed to generatee rng_keys")
     
     # MCMC
     parser.add_argument("--obs-idx", default=[54, 60, 100],
                         type=list,
                         help="index of observed locations")

     parser.add_argument("--mcmc-args", default={"num_samples": 10000, 
                                                  "num_warmup": 10000,
                                                  "num_chains": 4,
                                                  "thinning": 5}, 
                         type=dict)

     args = parser.parse_args()
     return args



if __name__ == "__main__":

     
     # use jitter=1.0e-5 for numerical stability
     # use noise=0.0 for training samples from GP
     args = args_parser()

     x = jnp.arange(0, 1, 1/args.n)
     poivae_mcmc(args, x, "poigp") 

     # gp_krig(args, x, "gp", 0.0)
     
     # varying lengthscales
     # lengths_list = [3.1, 0.05, 0.27]

     # vae_mcmc_ls(args, x, lengths_list, dat_noise=0.001)

     # # context points are 1, 10, 100, 200
     obs_idx_dict = {}
     # # obs_idx_dict['1'] = [100]
     obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
     # # obs_idx_dict['100'] = list(rd.sample(np.arange(args.n).tolist(), k=100))
     # # obs_idx_dict['200'] = list(rd.sample(np.arange(args.n).tolist(), k=200))

     rng_key = random.PRNGKey(args.seed)

     example_poigp(rng_key, x, 
     jitter=1.0e-5,
     plot_y=True,
     num_samp=10
     )

