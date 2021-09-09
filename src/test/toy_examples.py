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
from model.inference import Inference, VAEInference, GPInference


# y = x^3 + noise
def example_cubic(rng_key, x, noise=0, plot_y=True):
     rng_key, rng_key_y = random.split(rng_key)

     n = len(x)

     y = jnp.power(x, 3) + random.normal(rng_key_y, shape=(n, )) * noise

     ground_truth = {"x":np.arange(0,1,0.001), 
                    "y": np.arange(0,1,0.001)**3}

     return ground_truth, y

# trig function
def example_trig(rng_key, x, noise=0, plot_y=True):

     def func(x):
          return (
               np.sin(x * 3 * 3.14) 
               + 0.3 * np.cos(x * 9 * 3.14) 
               + 0.5 * np.sin(x * 7 * 3.14))

     rng_key, rng_key_y = random.split(rng_key)

     n = len(x)
     # mask = jnp.zeros(n, dtype=bool).at[obs_idx].set(True)
     # unobs_idx = jnp.arange(n)[~mask]

     y = func(x) + random.normal(rng_key_y, shape=(n, )) * noise
     # y_filtered = ops.index_update(y, unobs_idx, np.NaN)

     ground_truth = {"x":np.arange(0,1,0.001), 
                    "y": func(np.arange(0,1,0.001))}
     # if plot_y:
     #      plt.figure()
     #      plt.plot(ground_truth["x"], 
     #                ground_truth["y"], c="lightgray")
     #      plt.scatter(x, y, c="orange", label="unobserved")
     #      # plt.scatter(x, y_filtered, color="orange", label="observed")
     #      plt.legend(loc="upper left")  
     #      plt.show()
     #      plt.close()

     return ground_truth, y

# draws from GP
def example_gp(
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
     gp_y = GP(kernel=exp_kernel2, jitter=jitter, d=1)
     y = Predictive(gp_y.sample, num_samples=num_samp)(rng_key=rng_key, x=x, ls=ls, var=var, sigma=noise)["y"]
     
     ground_truth = {"x": x, 
                    "y": y[0]}
     
     if plot_y:
          plt.figure()
          for i in range(num_samp):
               # plt.plot(ground_truth["x"], 
               #           ground_truth["y"][i], c="lightgray")
               plt.plot(x, y[i])
          # plt.scatter(x, y_filtered, color="orange", label="observed")
          # plt.legend(loc="upper left")  
          plt.show()
          plt.close()

     return ground_truth, y[0]
     
def vae_mcmc(args, x, example, dat_noise=0):
     """ Plot posterior predictions for varying size of unserved locations
     with index ranging from 1, 10, 40, 100 for a total of 300 (default) 
     dense grid over [0,1].
     args: namespace, args of GP, VAE and Inference
     example: string, example function used to generate data
     dat_noise: float, added noise to generated data
     """

     assert args.d is 1
     
     # generate example data ------------------------------------------------
     if example == "cubic":
          example_func = example_cubic
     elif example == "trig":
          example_func = example_trig
     elif example == "gp":
          example_func = example_gp
     else:
          raise NotImplementedError

     # n points over [0,1]
     # x = jnp.arange(0, 1, 1/args.n)

     rng_key = random.PRNGKey(args.seed+1)
     # generate data for inference
     ground_truth, y = example_func(rng_key, x, noise=dat_noise, plot_y=False)

     # setup GP ---------------------------------------------------------------
     if args.kernel == "exponential2":
          kernel = exp_kernel2
     # elif args.kernel == "exponential1":
     #      kernel = exp_kernel1
     else:
          raise NotImplementedError
     # ls, var and noise are random in training
     gp = GP(
          kernel=kernel, 
          jitter=args.jitter, 
          d=args.d)
     
     # VAE training -----------------------------------------------------------
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

     decoder = vae.vae_decoder()[1]
     # decoder_params = vae.fit(plot_loss=True)
     # Save decoder params
     # with open('src/test/decoders/decoder_parmas_1d_n300_GP', 'wb') as file:
     #      pickle.dump(decoder_params, file)

     # # open decoder params from file
     with open('src/test/decoders/decoder_parmas_1d_n300_GP_fixls', 'rb') as file:
          decoder_params = pickle.load(file)

     # Inference --------------------------------------------------------------

     # initialise the Inference object
     inference = VAEInference(
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
     vae_draws = vae_predictive(rng_key_predict)['y_pred']

     plt.figure()
     for i in range(10):
          plt.plot(x, vae_draws[i])
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
     plt.savefig('src/test/plots/vae_mcmc_{}_fixls.png'.format(example))
     plt.show()
     plt.close()



def vae_mcmc_ls(args, x, lengths_list, var=0.5, dat_noise=0):
     """ Plot posterior predictions for GP examples varying lengthscales 
     [2.96, 0.05, 0.27].
     args: namespace, args of GP, VAE and Inference
     example: string, example function used to generate data
     dat_noise: float, added noise to generated data
     """

     assert args.d is 1

     # n points over [0,1]
     # x = jnp.arange(0, 1, 1/args.n) # change it to input

     rng_key = random.PRNGKey(args.seed)
     # observation locations
     obs_idx = jnp.asarray(list(rd.sample(np.arange(args.n).tolist(), k=20)))

     gp_y = GP(kernel=exp_kernel2, jitter=args.jitter, d=1)
     y_Predictive = Predictive(gp_y.sample, num_samples=1)
     y_ = partial(y_Predictive, x=x, var=var, sigma=dat_noise)

     # setup GP ---------------------------------------------------------------
     if args.kernel == "exponential2":
          kernel = exp_kernel2
     # elif args.kernel == "exponential1":
     #      kernel = exp_kernel1
     else:
          raise NotImplementedError
     # ls, var and noise are random in training
     gp = GP(
          kernel=kernel, 
          jitter=args.jitter, 
          d=args.d)
     
     # VAE training -----------------------------------------------------------
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

     decoder = vae.vae_decoder()[1]
     # decoder_params = vae.fit(plot_loss=True)
     # # Save decoder params
     # with open('src/test/decoders/decoder_parmas_1d_n300_GP', 'wb') as file:
     #      pickle.dump(decoder_params, file)

     # open decoder params from file
     with open('src/test/decoders/decoder_parmas_1d_n300_GP', 'rb') as file:
          decoder_params = pickle.load(file)

     # Inference --------------------------------------------------------------

     # VAE inference
     inference_vae = VAEInference(
          decoder, decoder_params,
          args.z_dim, args.d,
          x, None, # None for generated data, y_filtered
          args.obs_idx,
          args.mcmc_args,
          args.seed
          )
     # exact GP inference
     inference_gp = GPInference(
          kernel,
          args.d,
          x, None, # None for generated data, y_filtered
          args.obs_idx,
          args.mcmc_args,
          args.seed
          )

     # plot samples fromtrained VAE prior -------------------------------------
     vae_predictive = Predictive(inference_vae.regression, num_samples=10)
     rng_key, rng_key_predict = random.split(random.PRNGKey(1))
     vae_draws = vae_predictive(rng_key_predict)['y_pred']

     plt.figure()
     for i in range(10):
          plt.plot(x, vae_draws[i])
     plt.show()
     plt.close()

     # plot for varying lengthscales -----------------------------------------------------
     plt.figure(figsize=(19, 12.5))
     for i, ls in enumerate(lengths_list):
          print(i)
          print(ls)

          rng_key, rng_key_i = random.split(rng_key)
          # generate data for inference
          y = y_(rng_key=rng_key_i, ls=ls)["y"][0]
          ground_truth = {"x": x, 
                         "y": y}
          # ground_truth, y = example_gp(rng_key, x, ls=ls, noise=dat_noise, plot_y=False)
 
          # mask unobserved y
          mask = jnp.zeros(args.n, dtype=bool).at[obs_idx].set(True)
          unobs_idx = jnp.arange(args.n)[~mask]
          y_filtered = ops.index_update(y, ops.index[unobs_idx], np.nan)

          # update observation locations, y and ground truth
          inference_vae.obs_idx = obs_idx
          inference_vae.y = y_filtered
          inference_vae.ground_truth = ground_truth

          inference_gp.obs_idx = obs_idx
          inference_gp.y = y_filtered
          inference_gp.ground_truth = ground_truth

          # obtain predictions
          _, pred_vae = inference_vae.fit(plot=False)
          _, pred_gp = inference_gp.gp_fit(plot=False)

          ## posterior
          plt.subplot(2,3,i+1)
          inference_vae.plot_prediction(pred_gp, x=jnp.delete(x, obs_idx))
          plt.title('GP-lengthscale-{}'.format(ls))

          plt.subplot(2,3,i+4)
          inference_vae.plot_prediction(pred_vae)
          plt.title('VAE-lengthscale-{}'.format(ls))
     
     plt.tight_layout()
     plt.savefig('src/test/plots/gp_vae_ls_1d.png')
     plt.show()
     plt.close()

# def gp_mcmc(args, example, dat_noise=0):

#      assert args.d is 1

#      if args.kernel == "exponential2":
#           kernel = exp_kernel2
#      # elif args.kernel == "exponential1":
#      #      kernel = exp_kernel1
#      else:
#           raise NotImplementedError
     
#      if example == "cubic":
#           example_func = example_cubic
#      elif example == "trig":
#           example_func = example_trig
#      elif example == "gp":
#           example_func = example_gp
#      else:
#           raise NotImplementedError

#      # n points over [0,1]
#      x = jnp.arange(0, 1, 1/args.n)

#      # context points are 1, 10, 100, 200
#      obs_idx_dict = {}
#      # obs_idx_dict['1'] = [100]
#      obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
#      # obs_idx_dict['100'] = list(rd.sample(np.arange(args.n).tolist(), k=100))
#      # obs_idx_dict['200'] = list(rd.sample(np.arange(args.n).tolist(), k=200))

#      rng_key = random.PRNGKey(args.seed+1)

#      # generate data for inference
#      ground_truth, y = example_func(rng_key, args.n, x, sigma=dat_noise)

     
#      inference = GPInference(
#           kernel,
#           args.d,
#           x, None, # None for generated data, y_filtered
#           args.obs_idx,
#           args.mcmc_args,
#           args.seed,
#           ground_truth
#           )

#      plt.figure(figsize=(15, 12.5))

#      # Inference
#      for i, item in enumerate(obs_idx_dict.items()):
#           print(i)
#           k, obs_idx = item
#           obs_idx = jnp.asarray(obs_idx)

#           # mask unobserved y
#           mask = jnp.zeros(args.n, dtype=bool).at[obs_idx].set(True)
#           unobs_idx = jnp.arange(args.n)[~mask]
#           y_filtered = ops.index_update(y, unobs_idx, np.NaN)

#           # update observation locations
#           inference.obs_idx = obs_idx
#           inference.y = y_filtered

#           # obtain predictions
#           prior_pred, pred = inference.fit(plot=False)

#           ## posterior
#           plt.subplot(2,1,i+1)
#           inference.plot_prediction(prior_pred)
#           plt.title('Posterior-'+k)

#           plt.subplot(2,1,i+2)
#           inference.plot_prediction(pred)
#           plt.title('Posterior-'+k)
     
#      plt.tight_layout()
#      # plt.savefig('src/test/plots/gp_mcmc_{}.png'.format(example))
#      plt.show()
#      plt.close()


def gp_krig(args, x, example, dat_noise=0):
     assert args.d is 1

     if args.kernel == "exponential2":
          kernel = exp_kernel2
     # elif args.kernel == "exponential1":
     #      kernel = exp_kernel1
     else:
          raise NotImplementedError
     
     if example == "cubic":
          example_func = example_cubic
     elif example == "trig":
          example_func = example_trig
     elif example == "gp":
          example_func = example_gp
     else:
          raise NotImplementedError

     # n points over [0,1]
     # x = jnp.arange(0, 1, 1/args.n)
     rng_key = random.PRNGKey(args.seed+1)
     # generate data for inference
     ground_truth, y = example_func(rng_key, x=x, noise=dat_noise)

     # context points are 1, 10, 40, 100
     obs_idx_dict = {}
     obs_idx_dict['1'] = [200]
     obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
     obs_idx_dict['40'] = list(rd.sample(np.arange(args.n).tolist(), k=40))
     obs_idx_dict['100'] = list(rd.sample(np.arange(args.n).tolist(), k=100))
     
     # do inference
     inference = GPInference(
          kernel,
          args.d,
          x, None, # None for generated data, y_filtered
          args.obs_idx,
          args.mcmc_args,
          args.seed,
          ground_truth
          )
     
     plt.figure(figsize=(15, 12.5))

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

          # obtain predictions and update 
          _, pred = inference.gp_fit(plot=False)
         
          ## posterior
          # plt.subplot(2,1,i+1)
          # inference.plot_prediction(prior_pred)
          # plt.title('Posterior-'+k)
          plt.subplot(2,2,i+1)
          inference.plot_prediction(pred, x=jnp.delete(x, obs_idx))
          plt.title('Posterior-'+k)
     
     plt.tight_layout()
     plt.savefig('src/test/plots/gp_krig_{}.png'.format(example))
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
     parser.add_argument("--jitter", default=1.0e-5, 
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
     parser.add_argument("--num-epochs", default=50, 
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

     parser.add_argument("--mcmc-args", default={"num_samples": 1000, 
                                                  "num_warmup": 1000,
                                                  "num_chains": 4,
                                                  "thinning": 3}, 
                         type=dict)

     args = parser.parse_args()
     return args



if __name__ == "__main__":

     
     # use jitter=1.0e-5 for numerical stability
     # use noise=0.0 for training samples from GP
     args = args_parser()

     x = jnp.arange(0, 1, 1/args.n)
     vae_mcmc(args, x, "gp", dat_noise=0.0) 

     # gp_krig(args, x, "gp", 0.0)
     
     # varying lengthscales
     lengths_list = [2.96, 0.05, 0.27]

     # vae_mcmc_ls(args, x, lengths_list, dat_noise=0.001)

     # # context points are 1, 10, 100, 200
     obs_idx_dict = {}
     # # obs_idx_dict['1'] = [100]
     obs_idx_dict['10'] = [20, 23, 100, 110, 117, 130, 133, 140, 170, 190]
     # # obs_idx_dict['100'] = list(rd.sample(np.arange(args.n).tolist(), k=100))
     # # obs_idx_dict['200'] = list(rd.sample(np.arange(args.n).tolist(), k=200))

     rng_key = random.PRNGKey(args.seed)

     # example_gp(rng_key, x, 
     # jitter=1.0e-5,
     # plot_y=True,
     # num_samp=10
     # )
