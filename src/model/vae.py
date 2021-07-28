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

class VAE:
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
          # y, 
          # ground_truth,
          # obs_idx,
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
          self.num_test = num_train
          self.x = x
          self.rng_key = random.PRNGKey(seed)
          
          # Predictive function for sampling GP training set
          self.gp_predictive = None
          self.svi = None

          # self.y = y 
          # self.ground_truth = ground_truth
          # self.obs_idx = obs_idx          

     
     # for loop within stax.serial? -- general way of stacking?
     def vae_encoder(self):
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
          return stax.serial(
               stax.Dense(self.hidden_dims[1], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.hidden_dims[0], W_init=stax.randn()),
               stax.Relu,
               stax.Dense(self.out_dim, W_init=stax.randn()) 
          )
     
     def vae_model(self, batch):
          batch = jnp.reshape(batch, (batch.shape[0], -1))
          decode = numpyro.module(
               "decoder", 
               self.vae_decoder(), 
               (self.batch_size, self.z_dim))
          z = numpyro.sample(
               "z", 
               dist.Normal(jnp.zeros((self.z_dim,)), jnp.ones((self.z_dim,))))
          gen_loc = decode(z)

          return numpyro.sample("obs", dist.Normal(gen_loc, .1), obs=batch) 


     def vae_guide(self, batch):
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

          def body_fn(i, val):
               rng_key_i = random.fold_in(rng_key, i)
               rng_key_i, rng_key_ls = random.split(rng_key_i)

               loss_sum, svi_state = val # val -- svi_state
               # directly draw sample from the GP for a random lengthscale
               length_i = numpyro.sample("length", dist.Beta(0.2, 1.0), rng_key=rng_key_ls)
               batch = self.gp_predictive(rng_key_i, length_i, self.x)

               # `update` returns (svi_state, loss)
               svi_state, loss = self.svi.update(svi_state, batch['y']) 
               loss_sum += loss # / self.batch_size
               return loss_sum, svi_state

          return lax.fori_loop(0, self.num_train, body_fn, (0.0, svi_state))
     
     @partial(jit, static_argnums=(0,))
     def eval_test(self, rng_key, svi_state):

          def body_fn(i, loss_sum):
               rng_key_i = random.fold_in(rng_key, i) 
               rng_key_i, rng_key_ls = random.split(rng_key_i)
               
               length_i = numpyro.sample("length", dist.Beta(0.2, 1.0), rng_key=rng_key_ls)
               batch = self.gp_predictive(rng_key_i, length_i, self.x)

               loss = self.svi.evaluate(svi_state, batch['y']) / self.batch_size
               loss_sum += loss
               return loss_sum

          loss = lax.fori_loop(0, self.num_test, body_fn, 0.0)
          loss = loss / self.num_test

          return loss
     
     def fit(self, plot_loss=True):
          adam = optim.Adam(self.learning_rate)
          self.svi = SVI(
               self.vae_model, 
               self.vae_guide, 
               adam, 
               Trace_ELBO()
          )
          encoder_nn = self.vae_encoder()
          decoder_nn = self.vae_decoder()
          rng_key, rng_key_samp, rng_key_init = random.split(self.rng_key, 3)

          self.gp_predictive = Predictive(self.gp.sample, num_samples=self.batch_size)

          # initialise with a sample batch
          sample_batch = self.gp_predictive(rng_key=rng_key_samp, ls=0.1, x=self.x)
          # debug why VAE is not training properly
          
          svi_state = self.svi.init(rng_key_init, sample_batch['y'])
          # print(svi_state)
          test_loss_list = []

          for i in range(self.num_epochs):
               rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)
               t_start = time.time()
               # num_train, train_idx = train_init() 

               _, svi_state = self.epoch_train(rng_key_train, svi_state)
               test_loss = self.eval_test(rng_key_test, svi_state)
               test_loss_list += [test_loss]

               print(
                    "Epoch {}: loss = {} ({:.2f} s.)".format(
                         i, test_loss, time.time() - t_start
                    )
               )

          if plot_loss:
               plt.figure()
               plt.plot(np.arange(0, self.num_epochs, 1), test_loss_list)
               plt.xlabel("epochs")
               plt.ylabel("test error")
               plt.show()
               plt.close()

          # # make prerdiction for y with current decoder------------------------
          # self.fit(decoder_nn[1], self.svi.get_params(svi_state)["decoder$params"])

          
          # we can add prediction for sin function 
          
          return decoder_nn[1], self.svi.get_params(svi_state)["decoder$params"]
          # decoder and optimal parameters for decoder
          
