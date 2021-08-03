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
          raise NotImplementedError

     # @partial(jit, static_argnums=(0,))
     def run_mcmc(self, rng_key, y, obs_idx, model):
          numpyro.enable_validation(is_validate=True)
          start = time.time()
          # we may choose other ones, but for now -- `init_to_medians`
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
          
          mcmc.run(rng_key, y, obs_idx)
          mcmc.print_summary()
          print("\nMCMC elapsed time:", time.time() - start)

          return mcmc.get_samples()

     def fit(self, plot=True):
          rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(self.rng_key, 4)

          # we may want to check how the prediction does with prior
          prior_predictive = Predictive(self.regression, num_samples=1000)
          prior_predictions = prior_predictive(rng_key_prior)["y_pred"]

          # after training, we sample from the posterior 
          post_samples = self.run_mcmc(
               rng_key_post, 
               self.y, 
               self.obs_idx, 
               model=self.regression)
          # print(np.unique(np.isnan(post_samples['kernel_noise']), return_counts=True))
          # print(np.unique(np.isnan(post_samples['kernel_var']), return_counts=True))
          # print(np.unique(np.isnan(post_samples['kernel_length']), return_counts=True))
          # print(np.unique(np.isnan(post_samples['y_pred']), return_counts=True))
          # get samples from predictive distribution
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

     # some plots
     def plot_prediction(self, y_pred, x=None):
          if x is None:
               x = self.x
          # ground truth is a dictionaty storing dense grid for x and value of y
          mean_pred = jnp.mean(y_pred, axis=0)
          hpdi_pred = hpdi(y_pred, 0.9)
          # print(np.unique(np.isnan(y_pred), return_counts=True))
          # print(y_pred)
          # print(mean_pred)
          # plt.figure()
          for i in range(20):
               plt.plot(x, y_pred[i], color="lightgray", alpha=0.2)

          plt.fill_between(x.flatten(), 
                              hpdi_pred[0], hpdi_pred[1], 
                              alpha=0.4, interpolate=True)
          plt.plot(x, mean_pred, label="mean prediction")
          plt.scatter(self.x, self.y, c="orange")
          if self.ground_truth is not None:
               plt.plot(self.ground_truth["x"], 
                         self.ground_truth["y"], 
                         color="orange", label="ground truth")
          # plt.legend(loc="upper left")
     
     def plot_prediction2(self, y_pred):
          # this is for plotting grid only!
          mean_pred = jnp.mean(y_pred, axis=0)
          # hpdi_pred = hpdi(y_, 0.9)
          # diff = self.x[0, 1] - self.x[0, 0]
          plt.scatter(
               self.x[self.obs_idx, 0]+1/(2*self.n), 
               self.x[self.obs_idx, 1]+1/(2*self.n), 
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

          noise = numpyro.sample("kernel_noise", dist.Beta(0.6, 2.0))
          z = numpyro.sample("z", 
                              dist.Normal(jnp.zeros(self.z_dim), jnp.ones(self.z_dim)))

          f = numpyro.deterministic("f", self.decoder(self.decoder_params, z))

          if y is None: # durinig prediction
               numpyro.sample("y_pred", dist.Normal(f, noise))
          else: # during inference
               numpyro.sample(
                    "y", 
                    dist.Normal(f[obs_idx], noise), 
                    obs=y[obs_idx])


class GPInference(Inference):
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
          # set uninformative log-normal priors on three kernel hyperparameters
          # what is used in the GP inference numpyro example
          var = numpyro.sample("kernel_var", dist.Beta(0.8, 1.5))
          noise = numpyro.sample("kernel_noise", dist.Beta(0.6, 2.0))
          # var = 1
          # noise = 0.001
          ls = numpyro.sample("kernel_length", dist.Beta(0.8, 2.0))

          # compute kernel (plus noise, i.e. nugget)
          k = self.kernel(self.x, self.x, self.d, var, ls, noise=0, jitter=1.0e-4)
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
                    dist.Normal(f, noise))
               # numpyro.deterministic("y_pred", f)
                    
          else: # during inference
               numpyro.sample(
                    "y", 
                    dist.Normal(f[obs_idx], noise),
                    obs=y[obs_idx])
               # numpyro.deterministic("y", f[obs_idx])


     def gp_regression(self, x, y=None):
          var = numpyro.sample("kernel_var", dist.Beta(0.8, 1.5))
          noise = numpyro.sample("kernel_noise", dist.Beta(0.6, 2.0))
          # var = 1
          # noise = 0.001
          ls = numpyro.sample("kernel_length", dist.Beta(0.8, 2.0))

          # compute kernel
          k = self.kernel(x, x, self.d, var, ls, noise)

          # sample Y according to the standard gaussian process formula
          numpyro.sample(
               "y_pred",
               dist.MultivariateNormal(
                    loc=jnp.zeros(x.shape[0]), 
                    covariance_matrix=k),
               obs=y
          )

     
     def gp_prediction(self, rng_key, x, y, var, ls, noise):
          # compute kernels between train and test data, etc.
          x_test = jnp.delete(x, self.obs_idx)
          x_obs = x[self.obs_idx]

          k_pp = self.kernel(x_test, x_test, self.d, var, ls, noise)
          k_px = self.kernel(x_test, x_obs, self.d, var, ls, noise, include_noise=False)
          k_xx = self.kernel(x_obs, x_obs, self.d, var, ls, noise)
          k_xx_inv = jnp.linalg.inv(k_xx)
          K = k_pp - jnp.matmul(k_px, jnp.matmul(k_xx_inv, jnp.transpose(k_px)))
          sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
               rng_key, x_test.shape[:1]
          )
          mean = jnp.matmul(k_px, jnp.matmul(k_xx_inv, y[self.obs_idx]))
          # return mean and samples
          print("shape of mean GP", x_test.shape)
          print(x_obs.shape)
          return mean, mean + sigma_noise
     

     def gp_run_mcmc(self, rng_key, x, y, model):
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
          rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(self.rng_key, 4)

          # we may want to check how the prediction does with prior
          # prior_pred = 
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
               post_samples["kernel_noise"],
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

               # plt.subplot(1,2,1)
               # plot_func(prior_predictions)
               # plt.title('Prior')

               # plt.subplot(1,2,2)
               plot_func(predictions, x=jnp.delete(self.x, self.obs_idx))
               plt.title('Posterior')

               # plt.savefig('.png', bbox_inches='tight')
               plt.show()
               plt.close()

          return None, predictions

