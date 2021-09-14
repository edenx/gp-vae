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
from model.gp import GP, exp_kernel2, agg_kernel_grid
from model.vae import VAE
from model.inference import VAEInference, GPInference


# draws from GP
def example_gp2(
     rng_key, 
     x, 
     ls=0.27, # ls=0.07,
     var=None, # var=0.5,
     noise=None, # sigma=0, 
     jitter=1.0e-5,
     plot_y=True,
     num_samp=1,
     columns=4
     ):
     # we use the default kwargs in GP apart from noise
     gp_y = GP(kernel=exp_kernel2, jitter=jitter, d=2)
     y = Predictive(gp_y.sample, num_samples=num_samp)(rng_key=rng_key, x=x, ls=ls, var=var, sigma=noise)["y"]
     
     ground_truth = {"x": x, 
                    "y": y[0]}
     
     if plot_y:
          n = int(jnp.power(x.shape[0], 1/2))
          plot_samps(y, n, num_samp, columns=columns)

     return ground_truth, y[0]
    

def plot_samps(y, n, num_samp, columns=4):

     if num_samp > 1:
          rows = ceil(num_samp / columns)

          fig, axs = plt.subplots(rows, columns, figsize=(19, 4*rows))
          _min, _max = np.amin(y[0:columns*rows]), np.amax(y[0:columns*rows])
          for r in range(rows):
               for c in range(columns):
                    im = axs[r, c].imshow(y[r*columns + c].reshape(n,n), 
                                   cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',
                                   vmin=_min, vmax=_max)
                    axs[r, c].set_title("draw "+str(r*columns + c))  
                    fig.colorbar(im, ax=axs[r, c])
          fig.suptitle("samples of GP", fontsize=16)
          plt.show()
          plt.close()
     else:
          plt.imshow(
               y.reshape(n, n), 
               alpha=0.7,
               cmap='viridis',
               interpolation='none', 
               extent=[0,1,0,1], # subject to change, choose e.g. x.min()
               origin='lower')      
          # plt.title("sample of GP", fontsize=16)    
          plt.colorbar()
          plt.close()



def vae_mcmc(args, x, example, dat_noise=0.002):
     """ Plot posterior predictions for varying size of unserved locations
     with index ranging from 1, 10, 100, 200 for a total of 300 (default) 
     dense grid over [0,1].
     args: namespace, args of GP, VAE and Inference
     example: string, example function used to generate data
     dat_noise: float, added noise to generated data"""

     assert args.d is 2
     
     if example == "gp2":
          example_func = example_gp2
     else:
          raise NotImplementedError

     # generate data for inference
     rng_key = random.PRNGKey(args.seed+1)
     
     ground_truth, y = example_func(rng_key, x, noise=dat_noise)

     if args.kernel == "exponential2":
          kernel = exp_kernel2
     else:
          raise NotImplementedError

     # VAE training
     gp = GP(
          kernel=kernel, 
          jitter=args.jitter, 
          d=args.d) # ls is random in training

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

     decoder = vae.vae_decoder()[1]
     # # Save decoder params
     # decoder_params = vae.fit(plot_loss=True)
     # with open('src/test/decoders/decoder_parmas_2d_n25_GP_fixls', 'wb') as file:
     #      pickle.dump(decoder_params, file)

     # # open decoder params from file
     with open('src/test/decoders/decoder_parmas_2d_n25_GP_fixls', 'rb') as file:
          decoder_params = pickle.load(file)

     # context points are 10, 40, 135, 400
     obs_idx_dict = {}
     obs_idx_dict['10'] = list(rd.sample(np.arange(args.n**2).tolist(), k=10))
     obs_idx_dict['40'] = (obs_idx_dict['10'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=30)))
     obs_idx_dict['125'] = (obs_idx_dict['40'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=85)))
     obs_idx_dict['400'] = (obs_idx_dict['125'] 
                              + list(rd.sample(np.arange(args.n**2).tolist(), k=375)))

     # initialise the Inference object
     inference = VAEInference(
          decoder, decoder_params,
          args.z_dim, args.d, # n locs on each axis
          x, y, # None for generated data, y_filtered
          obs_idx_dict['400'], # just put something here first
          args.mcmc_args,
          args.seed,
          ground_truth
          )

     # plot samples fromtrained VAE prior ---------------------------------------
     vae_predictive = Predictive(inference.regression, num_samples=12)
     rng_key, rng_key_predict = random.split(random.PRNGKey(1))
     vae_draws = vae_predictive(rng_key_predict)['y_pred']

     plt.figure()
     plot_samps(vae_draws, args.n, num_samp=12)
     plt.show()
     plt.close()

     # plot ground truth ---------------------------------------------------------
     plt.figure(figsize=(15/2, 25/4))
     inference.plot_prediction2(y[None, :])
     plt.title('Ground truth')
     plt.show()
     # plt.savefig('src/test/plots/test_example_gp2_ground_truth.png'.format(example))
     plt.close()

     # Inference ------------------------------------------------------------------
     fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(18,8))
     axs[0,0].imshow(y[None, :].reshape((args.n, args.n)), 
                    cmap='viridis', interpolation='none', 
                    extent=[0,1,0,1], origin='lower')
     axs[0,0].set_title("ground truth")  

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
          pred_mean = jnp.mean(pred, axis=0)
          pred_std = jnp.std(pred, axis=0)
          # posterior
          im = axs[0,i+1].imshow(pred_mean.reshape((args.n, args.n)),
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[0,i+1].set_title("inferred mean: " + str(k + " points"))

          im = axs[1,i+1].imshow(pred_std.reshape((args.n, args.n)),
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[1,i+1].set_title("inferred std: " + str(k + " points")) 
     
     axs[1,0].axis('off')
     cbaxes = fig.add_axes([0.13, 0.14, 0.01, 0.31])
     fig.colorbar(im, ax=axs[1,0], cax=cbaxes)

     plt.tight_layout()
     plt.savefig('src/test/plots/vae_mcmc_{}2_fixls.png'.format(example))
     plt.show()
     plt.close()




def vae_mcmc_ls(args, x, lengths_list, var=0.5, dat_noise=0):
     """ Plot posterior predictions for GP examples varying lengthscales 
     [2.96, 0.05, 0.27].
     args: namespace, args of GP, VAE and Inference
     example: string, example function used to generate data
     dat_noise: float, added noise to generated data
     """

     assert args.d is 2

     # n points over [0,1]
     # x = jnp.arange(0, 1, 1/args.n) # change it to input

     rng_key = random.PRNGKey(args.seed)
     # 40 observation locations
     obs_idx = jnp.asarray(list(rd.sample(np.arange(args.n**2).tolist(), k=40)))

     gp_y = GP(kernel=exp_kernel2, jitter=args.jitter, d=2)
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
     # with open('src/test/decoders/decoder_parmas_1d_n300_GP_ep70', 'wb') as file:
     #      pickle.dump(decoder_params, file)

     # open decoder params from file
     with open('src/test/decoders/decoder_parmas_2d_n25_GP_fixls', 'rb') as file:
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
     # # exact GP inference
     # inference_gp = GPInference(
     #      kernel,
     #      args.d,
     #      x, None, # None for generated data, y_filtered
     #      args.obs_idx,
     #      args.mcmc_args,
     #      args.seed
     #      )

     # plot samples from trained VAE prior --------------------------------------------
     vae_predictive = Predictive(inference_vae.regression, num_samples=12)
     rng_key, rng_key_predict = random.split(random.PRNGKey(1))
     vae_draws = vae_predictive(rng_key_predict)['y_pred']

     plt.figure()
     plot_samps(vae_draws, args.n, num_samp=12)
     plt.show()
     plt.close()

     # plot for varying lengthscales -----------------------------------------------------
     # plt.figure(figsize=(19, 12.5))
     fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(19,16))
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

          # inference_gp.obs_idx = obs_idx
          # inference_gp.y = y_filtered
          # inference_gp.ground_truth = ground_truth

          # obtain predictions
          _, pred_vae = inference_vae.fit(plot=False)
          # _, pred_gp = inference_gp.gp_fit(plot=False)
          pred_vae_mean = jnp.mean(pred_vae, axis=0)
          pred_vae_std = jnp.std(pred_vae, axis=0)

          axs[i,0].axis('off')
          # ground truth
          im = axs[0,i+1].imshow(y.reshape((args.n, args.n)), 
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[0,i+1].scatter(
                    x[obs_idx, 0]+1/(2*args.n), 
                    x[obs_idx, 1]+1/(2*args.n), 
                    c=y[obs_idx])
          axs[0,i+1].set_title('GroundTruth ls: {}'.format(ls)) 

          cbaxes = fig.add_axes([0.13, 0.14, 0.01, 0.31])
          fig.colorbar(im, ax=axs[i,0], cax = cbaxes)

          # prediction mean
          im = axs[1,i+1].imshow(pred_vae_mean.reshape((args.n, args.n)), 
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[1,i+1].set_title('PredictionMean ls: {}'.format(ls))

          cbaxes = fig.add_axes([0.13, 0.14, 0.01, 0.31])
          fig.colorbar(im, ax=axs[i,0], cax = cbaxes)

          # prediction std
          im = axs[2,i+1].imshow(pred_vae_std.reshape((args.n, args.n)), 
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[2,i+1].set_title('PredictionStd ls: {}'.format(ls))    
     
          # axs[2,0].axis('off')
          cbaxes = fig.add_axes([0.13, 0.14, 0.01, 0.31])
          fig.colorbar(im, ax=axs[i,0], cax = cbaxes)

     plt.tight_layout()
     plt.savefig('src/test/plots/gp_vae_ls_2d_fixls.png')
     plt.show()
     plt.close()



def gp_krig(args, x, example, dat_noise=0.002):
     assert args.d is 2

     if args.kernel == "exponential2":
          kernel = exp_kernel2
     else:
          raise NotImplementedError
     
     if example == "gp2":
          example_func = example_gp2
     else:
          raise NotImplementedError

     # generate data for inference
     rng_key = random.PRNGKey(args.seed+1)
     ground_truth, y = example_func(rng_key, x, noise=dat_noise, plot_y=False)

     # context points are 10, 40, 135, 400
     obs_idx_dict = {}
     obs_idx_dict['10'] = list(rd.sample(np.arange(args.n**2).tolist(), k=10))
     # obs_idx_dict['40'] = (obs_idx_dict['10'] 
     #                          + list(rd.sample(np.arange(args.n**2).tolist(), k=30)))
     # obs_idx_dict['125'] = (obs_idx_dict['40'] 
     #                          + list(rd.sample(np.arange(args.n**2).tolist(), k=85)))
     # obs_idx_dict['400'] = (obs_idx_dict['125'] 
     #                          + list(rd.sample(np.arange(args.n**2).tolist(), k=375)))
     
     print("what is going on")
     # do inference
     inference = GPInference(
          kernel,
          args.d,
          x, None, # None for generated data, y_filtered
          obs_idx_dict['10'],
          args.mcmc_args,
          args.seed,
          ground_truth
          )
     
     fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(18,8))
     axs[0,0].imshow(y[None, :].reshape((args.n, args.n)), 
                    cmap='viridis', interpolation='none', 
                    extent=[0,1,0,1], origin='lower')
     axs[0,0].set_title("ground truth")  

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
          _, pred = inference.gp_fit(plot=False)
          print("prediction has shape", pred.shape)
          pred_mean = jnp.mean(pred, axis=0)
          pred_std = jnp.std(pred, axis=0)

          pred_mean_ = y_filtered
          pred_mean_ = ops.index_update(pred_mean_, unobs_idx, pred_mean)
          
          ## posterior
          # plt.subplot(2,1,i+1)
          # inference.plot_prediction(prior_pred)
          # plt.title('Posterior-'+k)
          # plt.subplot(2,2,i+1)
          # inference.plot_prediction2(pred_mean_, x=jnp.delete(x, obs_idx, axis=0))
          # plt.title('Posterior-'+k)

          # posterior
          im = axs[0,i+1].imshow(pred_mean_.reshape((args.n, args.n)),
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[0,i+1].set_title("inferred mean: " + str(k + " points"))

          im = axs[1,i+1].imshow(pred_std.reshape((args.n, args.n)),
                              cmap='viridis', interpolation='none', 
                              extent=[0,1,0,1], origin='lower')
          axs[1,i+1].set_title("inferred std: " + str(k + " points")) 
     
     axs[1,0].axis('off')
     cbaxes = fig.add_axes([0.13, 0.14, 0.01, 0.31])
     fig.colorbar(im, ax=axs[1,0], cax=cbaxes)
     
     plt.tight_layout()
     plt.savefig('src/test/plots/gp_krig_{}2_fixls.png'.format(example))
     plt.show()
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
     parser.add_argument("--ls", default=0.3, 
                         type=float, help="lengthscale of kernel")
     parser.add_argument("--noise", default=0.001, 
                         type=float, help="additional noise of GP")
     parser.add_argument("--jitter", default=1.0e-3, 
                         type=float, help="fixed additional noise to kernel diagonal for numerical stability")
     
     # VAE
     parser.add_argument("--n", default=25, 
                         type=int, help="number of point on grid")
     parser.add_argument("--hidden-dims", default=[50,30], 
                         type=list, help="dimension of hidden layers for encoder/decoder")
     parser.add_argument("--z-dim", default=15, 
                         type=int, help="bottleneck dimension")
     parser.add_argument("--batch-size", default=152, 
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

     parser.add_argument("--mcmc-args", default={"num_samples": 10000, 
                                                  "num_warmup": 10000,
                                                  "num_chains": 4,
                                                  "thinning": 3}, 
                         type=dict)

     args = parser.parse_args()
     return args


if __name__ == "__main__":

     # n = 25 # default in arg parser
     # d = 2 
     # grid = jnp.arange(0, 1, 1/n)
     

     # if d==1:
     #      x = grid
     #      # print(x.shape)
     # elif d==2:
     #      u, v = jnp.meshgrid(grid, grid)
     #      x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
     # else:
     #      raise NotImplementedError

     rng_key = random.PRNGKey(0)

     args = args_parser()

     # n by n grid over [0,1]x[0,1]
     grid = jnp.arange(0, 1, 1/args.n)
     u, v = jnp.meshgrid(grid, grid)
     x = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
     # print(x)
     # example_gp2(rng_key, x, num_samp=12)

     # vae_mcmc(args, x=x, example="gp2")

     # gp_krig(args, x=x, example="gp2")

     # varying lengthscales
     lengths_list = [3.1, 0.05, 0.3]

     vae_mcmc_ls(args, x, lengths_list, dat_noise=0.002)
