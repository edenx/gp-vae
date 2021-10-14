from pickle import FALSE
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops, vmap

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive
from numpyro.diagnostics import hpdi

# this is separate from VAE training
class Inference():
     """Class for inference methods.

     Attributes:
          d (int)             - spatial dimension.
          x (ndarray)         - spatial locations.
          y (ndarray)         - (function) value at x.
          obs_idx (list)      - index for observation location.
          mcmc_args (dict)    - dictionary containing parameters for MCMC.
          seed (int)          - random seed.
          ground_truth (dict) - ground truth {'x': x, 'y': y} (for highr resolution x and y)
          grid (bool)         - if True the data is supported on regular grid.
     """
     def __init__(
          self, 
          d, # spatial dim
          x, y,
          obs_idx, # indices for observed location
          mcmc_args, # dictionary
          seed=0,
          ground_truth=None,
          grid=True
          ):

          self.d = d
          self.x = x
          self.y = y
          
          self.obs_idx = jnp.asarray(obs_idx)
          self.mcmc_args = mcmc_args
          self.rng_key = random.PRNGKey(seed)
          self.ground_truth = ground_truth
          self.grid = grid

          if grid:
               self.n = int(jnp.power(x.shape[0], 1/d)) # only for d dim grid
          else:
               self.n = x.shape[0]
          
     
     # @partial(jit, static_argnums=(0,)) # leakage in tracer -- how to fix?
     def regression(self, y=None, obs_idx=None):
          """Model regression y=f(x)+noise.
          """
          raise NotImplementedError

     # @partial(jit, static_argnums=(0,))
     def run_mcmc(self, rng_key):
          """Run MCMC for given model and observations.

          Args:
               rng_key (ndarray) - a PRNGKey used as the random key.
               y (ndarray) - observations.
               obs_idx (ndarray) - location of observations.
               model - model for the inference.
          
          Returns:
               samples from posterior distribution.
          """
          numpyro.enable_validation(is_validate=True)
          start = time.time()
          # we may choose other ones, but for now -- `init_to_medians`
          init_strategy = init_to_median(num_samples=10)
          mcmc_kernel = NUTS(self.regression, init_strategy=init_strategy)
          mcmc = MCMC(
               mcmc_kernel,
               num_warmup=self.mcmc_args["num_warmup"],
               num_samples=self.mcmc_args["num_samples"],
               num_chains=self.mcmc_args["num_chains"],
               thinning=self.mcmc_args["thinning"],
               progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
          )

          mcmc.run(rng_key, y=self.y, obs_idx=self.obs_idx)
          mcmc.print_summary(exclude_deterministic=False)
          print("\nMCMC elapsed time:", time.time() - start)

          return mcmc.get_samples()

     def fit(self, plot=True):
          """Function to do MCMC on regression.

          Args:
               plot (bool) - if True plot the posterior prediction.
          
          Returns:
               prior and posterior predictions.
          """
          rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(self.rng_key, 4)

          # draws from prior
          prior_predictive = Predictive(self.regression, num_samples=1000)
          prior_predictions = prior_predictive(rng_key_prior)["y_pred"]
          # after training, we sample from the posterior 
          post_samples = self.run_mcmc(rng_key_post)

          # get samples from posterior predictive distribution
          predictive = Predictive(self.regression, post_samples)
          predictions = predictive(rng_key_pred)["y_pred"]
          # print(predictive(rng_key_pred))

          if self.d==1:
               plot_func = self.plot_prediction
          elif self.d==2 and self.grid:
               plot_func = self.plot_prediction2
          else: 
               raise NotImplementedError

          if plot:
               # print(prior_predictions)
               plt.figure(figsize=(12, 6))

               plt.subplot(1,2,1)
               plot_func(prior_predictions)
               plt.title('Prior')

               plt.subplot(1,2,2)
               plot_func(predictions)
               plt.title('Posterior')

               # plt.savefig('.png', bbox_inches='tight')
               plt.show()
               plt.close()

          return prior_predictions, predictions

     # some plots for posterior predictions or prior draws
     def plot_prediction(self, y_pred, x=None):
          """plot 1D posterior prediction.

          Args:  
               y_pred (ndarray) - prediction.
               x (ndarray) - spatial locations for y.
          """
          if x is None:
               x = self.x
          # ground truth is a dictionaty storing dense grid for x and value of y
          mean_pred = jnp.mean(y_pred, axis=0)
          hpdi_pred = hpdi(y_pred, 0.9)
          # print(np.unique(np.isnan(y_pred), return_counts=True))

          # plt.figure()
          for i in range(20):
               plt.plot(x, y_pred[i], color="lightgray", alpha=0.2)

          plt.fill_between(x, 
                              hpdi_pred[0], hpdi_pred[1], 
                              alpha=0.4, interpolate=True)
          plt.plot(x, mean_pred, label="mean prediction")
          plt.scatter(self.x, self.y, c="orange")
          if self.ground_truth is not None:
               plt.plot(self.ground_truth["x"], 
                         self.ground_truth["y"], 
                         color="orange", label="ground truth")
          # plt.legend(loc="upper left")
     
     def plot_prediction2(self, mean_pred, x=None):
          """plot 1D posterior prediction.

          Args:  
               mean_pred (ndarray) - mean of posterior prediction.
               x (ndarray) - spatial locations for y.
          """

          if x is None:
               x = self.x
          # this is for plotting grid only!
          # mean_pred = jnp.mean(pred, axis=0)
          
          # hpdi_pred = hpdi(y_, 0.9)
          # diff = self.x[0, 1] - self.x[0, 0]
          plt.scatter(
               x[self.obs_idx, 0]+1/(2*self.n), 
               x[self.obs_idx, 1]+1/(2*self.n), 
               c=self.y[self.obs_idx])
          plt.imshow(
               mean_pred.reshape((self.n, self.n)), 
               alpha=0.7,
               cmap='viridis',
               interpolation='none', 
               extent=[0,1,0,1], # subject to change, choose e.g. x.min()
               origin='lower')          
          plt.colorbar()
     


class VAEInference(Inference):
     """Class for MCMC inference with VAE trained prior.

     Attributes:
          decoder - decoder from VAE class.
          decoder_params - optimised parameters of decoder network.
          z_dim (int) - dimension of latent representation.
     """
     def __init__(
          self, 
          decoder, 
          decoder_params,
          z_dim, 
          *args,
          **kwargs,
          ):

          super().__init__(
               *args,
               **kwargs,
               )

          self.decoder = decoder
          self.decoder_params = decoder_params
          self.z_dim = z_dim


     def regression(self, y=None, obs_idx=None):
          """Regression function for MCMC.

          Args:
               y (ndarray) - function values (np.nan at unobserved locations).
               obs_idx (nd_array) - index of observation locations.
          """

          sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
          z = numpyro.sample("z", 
                              dist.Normal(jnp.zeros(self.z_dim), jnp.ones(self.z_dim)))
          f = numpyro.deterministic("f", self.decoder(self.decoder_params, z))

          if y is None: # durinig prediction
               numpyro.sample("y_pred", dist.Normal(f, sigma))
          else: # during inference
               numpyro.sample(
                    "y", 
                    dist.Normal(f[obs_idx], sigma), 
                    obs=y[obs_idx])


class PoiVAEInference(VAEInference):

     def fit(self):
          """Function to do MCMC on regression.

          Args:
               plot (bool) - if True plot the posterior prediction.
          
          Returns:
               prior and posterior predictions.
          """
          rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(self.rng_key, 4)

          # draws from prior
          prior_predictive = Predictive(self.regression, num_samples=1000)
          prior_predictions = prior_predictive(rng_key_prior)
          # after training, we sample from the posterior 
          post_samples = self.run_mcmc(rng_key_post)

          # get samples from posterior predictive distribution
          predictive = Predictive(self.regression, post_samples)
          predictions = predictive(rng_key_pred)
          # print(predictive(rng_key_pred))

          return prior_predictions, predictions

     def regression(self, y=None, obs_idx=None):
          """Regression function for MCMC.

          Args:
               y (ndarray) - function values (np.nan at unobserved locations).
               obs_idx (nd_array) - index of observation locations.
          """

          # sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
          z = numpyro.sample("z", 
                              dist.Normal(jnp.zeros(self.z_dim), jnp.ones(self.z_dim)))
          rate = numpyro.deterministic("rate", self.decoder(self.decoder_params, z))

          if y is None: # durinig prediction
               numpyro.sample("y_pred", dist.Poisson(rate))
          else: # during inference
               numpyro.sample(
                    "y", 
                    dist.Poisson(rate[obs_idx]), 
                    obs=y[obs_idx])


     def plot_prediction(self, pred, x=None):
          """plot 1D posterior prediction.

          Args:  
               y_pred (ndarray) - prediction.
               x (ndarray) - spatial locations for y.
          """
          if x is None:
               x = self.x

          y_pred = pred["y_pred"]
          rate_pred = pred["rate"]

          rate_mean =jnp.mean(rate_pred, axis=0)
          rate_hpdi =hpdi(rate_pred, 0.9)

          # print(np.unique(np.isnan(y_pred), return_counts=True))

          # plt.figure()

          plt.fill_between(x, 
                              rate_hpdi[0], rate_hpdi[1], 
                              alpha=0.4, interpolate=True)
          plt.plot(x, rate_mean, label="rate mean prediction")
          # plt.scatter(self.x, self.rate, c="red", alpha=0.7, label="")
          if self.ground_truth is not None:
               plt.plot(self.ground_truth["x"], 
                         self.ground_truth["rate"], label="ground truth rate")
               plt.scatter(self.ground_truth["x"], 
                         self.ground_truth["y"], 
                         color="orange", label="ground truth")


class GPInference(Inference):
     """Class for inference with GP prior.

     Attributes:
          kernel - kernel function.
     """
     def __init__(
          self, 
          kernel, 
          *args,
          **kwargs,
          ):

          super().__init__(
               *args,
               **kwargs,
               )

          self.kernel = kernel

     
     def regression(self, y=None, obs_idx=None):
          """Regression function for MCMC.

          Args:
               y (ndarray) - function values (np.nan at unobserved locations).
               obs_idx (nd_array) - index of observation locations.
          """
          # set uninformative log-normal priors on three kernel hyperparameters
          # what is used in the GP inference numpyro example
          var = numpyro.sample("kernel_var", dist.LogNormal(0.0,0.1))
          sigma = numpyro.sample("kernel_sigma", dist.HalfNormal(0.1))
          # var = 1
          # noise = 0.001
          ls = numpyro.sample("kernel_length",  dist.InverseGamma(4,1))

          # compute kernel (plus noise, i.e. nugget)
          k = self.kernel(self.x, self.x, var=var, ls=ls)
          # k = self.kernel(self.x, self.x, self.d, var, ls, noise=noise)

          f = numpyro.sample(
                    "f", 
                    dist.MultivariateNormal(
                         loc=jnp.zeros(self.x.shape[0]), 
                         covariance_matrix=k)
                    )
          
          # prediction without krigging
          if y is None: # durinig prediction
               numpyro.sample(
                    "y_pred", 
                    dist.Normal(f, sigma))
               # numpyro.deterministic("y_pred", f)
                    
          else: # during inference
               numpyro.sample(
                    "y", 
                    dist.Normal(f[obs_idx], sigma),
                    obs=y[obs_idx])
               # numpyro.deterministic("y", f[obs_idx])


     def gp_regression(self, x, y=None):
          """Kriging/exact GP regression.

          Args:
               x (ndarray) - observed spatial locations.
               y (ndarray) - observed values (not contain np.nan).
          """
          var = numpyro.sample("kernel_var", dist.LogNormal(0.0,0.1))
          sigma = numpyro.sample("kernel_sigma", dist.HalfNormal(0.1))
          # var = 1
          # noise = 0.001
          ls = numpyro.sample("kernel_length",  dist.InverseGamma(4,1))

          # compute kernel
          k = self.kernel(x, x, var=var, ls=ls)

          # sample Y according to the standard gaussian process formula
          f = numpyro.sample(
               "f",
               dist.MultivariateNormal(
                    loc=jnp.zeros(x.shape[0]), 
                    covariance_matrix=k)
          )

          numpyro.sample("y_pred", dist.Normal(f, sigma), obs=y)

     
     def gp_prediction(self, rng_key, x, y, var, ls, sigma):
          """Prediction using kriging.
          
          Args:
               rng_key (ndarray) - a PRNGKey used as the random key.
               x (ndarray) - (full) spatial locations.
               y (ndarray) - function values at x (np.nan at unobservred locations).
               var (float) - marginal variance of kernel for GP prior.
               ls (flaot) - lengthscale.
               sigma (flaot) - noise on the diagonal of Covariance matrix.

          Returns:
               posterior mean and posteerior samples.
          """
          # compute kernels between train and test data, etc.
          x_test = jnp.delete(x, self.obs_idx, axis=0)
          x_obs = x[self.obs_idx]

          print("shapee of x_test in prediction ", x_test.shape)
          print("shapee of x_train in prediction ", x_obs.shape)
          k_pp = self.kernel(x_test, x_test, var, ls)
          k_px = self.kernel(x_test, x_obs, var, ls)
          k_xx = self.kernel(x_obs, x_obs, var, ls)
          k_xx_inv = jnp.linalg.inv(k_xx + jnp.eye(len(self.obs_idx)) * sigma)
          K = k_pp - jnp.matmul(k_px, jnp.matmul(k_xx_inv, jnp.transpose(k_px)))
          noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
               rng_key, x_test.shape[:1]
          )
          mean = jnp.matmul(k_px, jnp.matmul(k_xx_inv, y[self.obs_idx]))
          # return mean and samples
          print("shape of mean GP", mean.shape)
          print(x_obs.shape)

          return mean, mean + noise
     

     def gp_run_mcmc(self, rng_key, x, y, model):
          """Draw samples from posterior with observed y.
          
          Args:
               rng_key (ndarray) - a PRNGKey used as the random key.
               x (ndarray) - observed spatial locations.
               y (ndarray) - observed values (not contain np.nan).
               model - model to be inffered.
          
          Returns:
               posteroir samples.
          """
          start = time.time()
          # demonstrate how to use different HMC initialization strategies
          init_strategy = init_to_median(num_samples=10)
          mcmc_kernel = NUTS(model, init_strategy=init_strategy)
          mcmc = MCMC(
               mcmc_kernel,
               num_warmup=self.mcmc_args["num_warmup"],
               num_samples=self.mcmc_args["num_samples"],
               num_chains=self.mcmc_args["num_chains"],
               thinning=self.mcmc_args["thinning"],
               progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
          )
          mcmc.run(rng_key, x, y)
          mcmc.print_summary()
          print("\nMCMC elapsed time:", time.time() - start)
          return mcmc.get_samples()
     

     def gp_fit(self, plot=True):
          """Get predictions with krigging.

          Returns:
               posterior prediciton at unobserved locations.
          """
          rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(self.rng_key, 4)

          # we may want to check prior samples
          prior_predictive = Predictive(self.regression, num_samples=1000)
          prior_predictions = prior_predictive(rng_key_prior)["y_pred"]

          # after training, we sample from the posterior 
          post_samples = self.gp_run_mcmc(
               rng_key_post, 
               self.x[self.obs_idx],
               self.y[self.obs_idx], 
               model=self.gp_regression)
          vmap_args = (
               random.split(rng_key_pred, post_samples["kernel_var"].shape[0]),
               post_samples["kernel_var"],
               post_samples["kernel_length"],
               post_samples["kernel_sigma"]
          )
          means, predictions = vmap(
               lambda rng_key, var, length, noise: self.gp_prediction(
                    rng_key, self.x, self.y, var, length, noise
               )
          )(*vmap_args)

          # mean_prediction = np.mean(means, axis=0)
          # percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
               
          if self.d==1:
               plot_func = self.plot_prediction
          elif self.d==2 and self.grid:
               plot_func = self.plot_prediction2
          else: 
               raise NotImplementedError

          if plot:
               # print(prior_predictions)
               plt.figure(figsize=(6, 6))

               plt.subplot(1,2,1)
               plot_func(prior_predictions)
               plt.title('Prior')

               plt.subplot(1,2,2)
               plot_func(predictions, x=jnp.delete(self.x, self.obs_idx, axis=0))
               plt.title('Posterior')

               # plt.savefig('.png', bbox_inches='tight')
               plt.show()
               plt.close()

          return means, predictions

