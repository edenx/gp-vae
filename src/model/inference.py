import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive
from numpyro.diagnostics import hpdi

# this is separate from VAE training
class Inference:
     def __init__(
          self, 
          decoder, 
          decoder_params,
          z_dim,
          x, y,
          obs_idx, # indices for observed location
          mcmc_args, # dictionary
          seed=0,
          ground_truth=None
          ):

          self.decoder = decoder
          self.decoder_params = decoder_params
          self.z_dim = z_dim
          self.x = x
          self.n = x.shape[0]
          self.obs_idx = jnp.asarray(obs_idx)
          self.mcmc_args = mcmc_args

          self.y = y
          self.rng_key = random.PRNGKey(seed)
          self.ground_truth = ground_truth
     
     # @partial(jit, static_argnums=(0,)) # leakage in tracer -- how to fix?
     def regression(self, y=None, obs_idx=None):

          sigma = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
          z = numpyro.sample("z", 
                              dist.Normal(jnp.zeros(self.z_dim), jnp.ones(self.z_dim)))

          f = numpyro.deterministic("f", self.decoder(self.decoder_params, z))

          if y is None: # durinig prediction
               numpyro.sample("y_pred", dist.Normal(f, sigma))
          else: # during inference
               numpyro.sample("y", 
                              dist.Normal(f[obs_idx], sigma), 
                              obs=y[obs_idx])

     # @partial(jit, static_argnums=(0,))
     def run_mcmc(self, rng_key, y, obs_idx, model):
          start = time.time()
          # we may choose other ones, but for now -- `init_to_medians`
          init_strategy = init_to_median(num_samples=10)
          kernel = NUTS(model, init_strategy=init_strategy)
          mcmc = MCMC(
               kernel,
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
          # print(post_samples)
          # get samples from predictive distribution
          predictive = Predictive(self.regression, post_samples)
          predictions = predictive(rng_key_pred)["y_pred"]
          # print(predictions)

          if plot:
               # print(prior_predictions)
               plt.figure(figsize=(12, 6))

               plt.subplot(1,2,1)
               self.plot_prediction(prior_predictions)
               plt.title('Prior')

               plt.subplot(1,2,2)
               self.plot_prediction(predictions)
               plt.title('Posterior')

               # plt.savefig('.png', bbox_inches='tight')
               plt.show()
               plt.close()

          return prior_predictions, predictions

     # some plots
     def plot_prediction(self, y_):
          # ground truth is a dictionaty storing dense grid for x and value of y
          mean_pred = jnp.mean(y_, axis=0)
          hpdi_pred = hpdi(y_, 0.9)

          # plt.figure()
          for i in range(20):
               plt.plot(self.x, y_[i], color="lightgray", alpha=0.1)

          plt.fill_between(self.x.flatten(), 
                              hpdi_pred[0], hpdi_pred[1], 
                              alpha=0.4, interpolate=True)
          plt.plot(self.x, mean_pred, label="mean prediction")
          plt.scatter(self.x, self.y, c="orange")
          if self.ground_truth is not None:
               plt.plot(self.ground_truth["x"], 
                         self.ground_truth["y"], 
                         color="orange", label="ground truth")
          
          # plt.legend(loc="upper left")
          # plt.show()
          # plt.savefig('.png', bbox_inches='tight')
          # plt.close()

 