import os
# general libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# JAX
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.experimental import stax

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.diagnostics import hpdi

# update the training to allow variance to be sampled along with ls
class VAE():
     """ 
     Implementation of VAE based on stax with 2 layer NN. 

     Attributes:
          gp             (cls) - GP class object.
          hidden_dims    (list) - dimension of hidden layers for encoder (symmetric to  decoder).
          z_dim          (int) - dimension of latent rerpresentation (bottleneck).
          out_dim        (int) - output dimension, usually same as input data dimension.
          batch_size     (int) - number of samples in a minibatch.
          num_epochs     (int) - number of training epochs.
          num_train      (int) - number of batches in training.
          num_test       (int) - number of batches in testing.
          learning_rate  (float) - learning rate for optimiser.
          x              (ndarrray) - spatial/temporal locations. 
          seed           (int) - random seed.
     """
     def __init__(
          self, 
          gp, # GP object
          hidden_dims, # list
          z_dim, # bottleneck
          out_dim,
          batch_size, 
          learning_rate,
          num_epochs,
          num_train,
          num_test,
          x,
          seed=0,
          ):
          
          self.gp = gp
          assert len(hidden_dims) == 2
          self.hidden_dims = hidden_dims # use reversed iterator for decoder
          self.z_dim = z_dim
          self.out_dim = out_dim
          self.batch_size = batch_size
          self.learning_rate = learning_rate
          self.num_epochs = num_epochs
          self.num_train = num_train
          self.num_test = num_test
          self.x = x
          self.rng_key = random.PRNGKey(seed)
          
          # Predictive function for sampling GP training sets
          self.gp_predictive = None
          self.svi = None       
     
     # for loop within stax.serial? -- general way of stacking?
     def vae_encoder(self):
          """Encoder network of VAE.
          """
          return stax.serial(
               stax.Dense(self.hidden_dims[0], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.hidden_dims[1], W_init=stax.randn()),
               stax.Relu,
               stax.FanOut(2),
               stax.parallel(
                    stax.Dense(self.z_dim, W_init=stax.randn()), # mean
                    stax.serial(stax.Dense(self.z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
               ),
          )

     def vae_decoder(self):
          """Decoder network of VAE.
          """
          return stax.serial(
               stax.Dense(self.hidden_dims[1], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.hidden_dims[0], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.out_dim, W_init=stax.randn()) 
          )
     
     def vae_model(self, batch):
          """Generation with decoder.

          Args: 
               batch (ndarray) - data batch.

          Returns:
               sample from pushed forward measure of z by decoder network.
          """
          batch = jnp.reshape(batch, (batch.shape[0], -1))
          decode = numpyro.module(
               "decoder", 
               self.vae_decoder(), 
               (self.batch_size, self.z_dim))
          z = numpyro.sample(
               "z", 
               dist.Normal(jnp.zeros((self.z_dim,)), jnp.ones((self.z_dim,))))
          f = decode(z)

          return numpyro.sample("obs", dist.Normal(f, .1), obs=batch) 


     def vae_guide(self, batch):
          """Inference with encoder.

          Args: 
               batch (ndarray) - data batch.

          Returns:
               sample from z.
          """
          batch = jnp.reshape(batch, (batch.shape[0], -1))
          encode = numpyro.module(
               "encoder", 
               self.vae_encoder(), 
               (self.batch_size, self.out_dim))
          z_loc, z_std = encode(batch)
          z = numpyro.sample("z", dist.Normal(z_loc, z_std))

          return z

     @partial(jit, static_argnums=(0,))
     def epoch_train(self, rng_key, svi_state):
          """Train VAE with samples drawn from GP with random lengthscale in each batch.

          Args:
               rng_key (ndarray) - a PRNGKey used as the random key.
               svi_state - current state of SVI.

          Returns:
               sum of ELBO loss and current state of SVI.
          """
          def body_fn(i, val):
               rng_key_i = random.fold_in(rng_key, i)
               rng_key_i, rng_key_ls, rng_key_var, rng_key_sigma = random.split(rng_key_i, 4)

               loss_sum, svi_state = val # val -- svi_state

               # directly draw sample from the GP for a random lengthscale
               length_i = numpyro.sample("length", dist.InverseGamma(1,.1), rng_key=rng_key_ls)
               var_i = numpyro.sample("var", dist.LogNormal(0,0.1), rng_key=rng_key_var)
               sigma_i = numpyro.sample("noise", dist.HalfNormal(0.1), rng_key=rng_key_sigma)
               batch = self.gp_predictive(rng_key_i, self.x
               , ls=length_i, var=var_i, sigma=sigma_i
               )

               # `update` returns (svi_state, loss)
               svi_state, loss = self.svi.update(svi_state, batch['y']) 
               loss_sum += loss # / self.batch_size
               return loss_sum, svi_state

          return lax.fori_loop(0, self.num_train, body_fn, (0.0, svi_state))
     
     # @partial(jit, static_argnums=(0,))
     def eval_test(self, rng_key, svi_state):
          """Test VAE with samples drawn from GP with random lengthscale in each batch.

          Args:
               rng_key (ndarray) - a PRNGKey used as the random key.
               svi_state - current state of SVI.

          Returns:
               average ELBO loss evaluated on test data sets.
          """
          def body_fn(i, loss_sum):
               rng_key_i = random.fold_in(rng_key, i) 
               rng_key_i, rng_key_ls, rng_key_var, rng_key_sigma = random.split(rng_key_i, 4)
               
               length_i = numpyro.sample("length", dist.InverseGamma(1,.1), rng_key=rng_key_ls)
               var_i = numpyro.sample("var", dist.LogNormal(0,0.1), rng_key=rng_key_var)
               sigma_i = numpyro.sample("noise", dist.HalfNormal(0.1), rng_key=rng_key_sigma)
 
               batch = self.gp_predictive(rng_key_i, self.x
               , ls=length_i, var=var_i, sigma=sigma_i
               )

               loss = self.svi.evaluate(svi_state, batch['y']) / self.batch_size
               loss_sum += loss
               return loss_sum

          loss = lax.fori_loop(0, self.num_test, body_fn, 0.0)
          loss = loss / self.num_test

          return loss
     
     def fit(self, plot_loss=True):
          """Train VAE with Adam optimiser and ELBO.

          Args:
               plot_loss (bool) - if True, plot the loss of test set each epoch

          Returns:
               Decoder network and optimised parameters of the decoder.
          """
          adam = optim.Adam(self.learning_rate)
          self.svi = SVI(
               self.vae_model, 
               self.vae_guide, 
               adam, 
               Trace_ELBO()
          )
          # encoder_nn = self.vae_encoder()
          # decoder_nn = self.vae_decoder()
          rng_key, rng_key_samp, rng_key_init = random.split(self.rng_key, 3)

          self.gp_predictive = Predictive(self.gp.sample, num_samples=self.batch_size)

          # initialise with a sample batch
          sample_batch = self.gp_predictive(rng_key=rng_key_samp, x=self.x)
          
          svi_state = self.svi.init(rng_key_init, sample_batch['y'])
          test_loss_list = []

          for i in range(self.num_epochs):
               rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)
               t_start = time.time()

               _, svi_state = self.epoch_train(rng_key_train, svi_state)
               test_loss = self.eval_test(rng_key_test, svi_state)
               test_loss_list += [test_loss]

               print(
                    "Epoch {}: loss = {} ({:.2f} s.)".format(
                         i, test_loss, time.time() - t_start
                    )
               )
          
               if np.isnan(test_loss): break

          if plot_loss:
               plt.figure()
               plt.plot(np.arange(0, self.num_epochs, 1)[0:len(test_loss_list)], test_loss_list)
               plt.xlabel("epochs")
               plt.ylabel("test error")
               plt.savefig('src/test/plots/vae_lost.png')
               plt.show()
               plt.close()

          # return optimal parameters for decoder
          return self.svi.get_params(svi_state)["decoder$params"]



class PoiVAE(VAE):

     def vae_decoder(self):
          """Decoder network of VAE.
          """
          return stax.serial(
               stax.Dense(self.hidden_dims[1], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.hidden_dims[0], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.out_dim, W_init=stax.randn()),
               stax.exp
          )

     def vae_model(self, batch):
          """Generation with decoder.

          Args: 
               batch (ndarray) - data batch.

          Returns:
               sample from pushed forward measure of z by decoder network.
          """
          batch = jnp.reshape(batch, (batch.shape[0], -1))
          decode = numpyro.module(
               "decoder", 
               self.vae_decoder(), 
               (self.batch_size, self.z_dim))
          z = numpyro.sample(
               "z", 
               dist.Normal(jnp.zeros((self.z_dim,)), jnp.ones((self.z_dim,))))
          rate = decode(z)

          return numpyro.sample("obs", dist.Poisson(rate), obs=batch) 