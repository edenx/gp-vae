
# general libraries
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# JAX
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.experimental import stax

from functools import partial

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive
from numpyro.diagnostics import hpdi

# assert numpyro.__version__.startswith('0.6.0')
@jit
def exp_kernel(x, z, var, length, noise, jitter=1.0e-6):

    deltaXsq = jnp.power((x[:, None] - z), 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq / length)
    k += (noise + jitter) * jnp.eye(x.shape[0])

    return k

def vae_encoder(hidden_dim1, hidden_dim2, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Relu,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()), # mean
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
        ),
    )


def vae_decoder(hidden_dim1, hidden_dim2, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(out_dim, W_init=stax.randn()) 
    )


def vae_model(batch, hidden_dim1=40, hidden_dim2=20, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", vae_decoder(hidden_dim1, hidden_dim2, out_dim), (batch_dim, z_dim))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
    gen_loc = decode(z)
    return numpyro.sample("obs", dist.Normal(gen_loc, .1), obs=batch) 


def vae_guide(batch, hidden_dim1=400, hidden_dim2=20, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", vae_encoder(hidden_dim1, hidden_dim2, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    z = numpyro.sample("z", dist.Normal(z_loc, z_std))
    return z


def main(args):

     adam = optim.Adam(step_size=1.0e-3)
     svi = SVI(
          vae_model, 
          vae_guide, 
          adam, 
          Trace_ELBO(), 
          hidden_dim1=args["hidden_dim1"], 
          hidden_dim2=args["hidden_dim2"], 
          z_dim=args["z_dim"]
     )
     encoder_nn = vae_encoder(args["hidden_dim1"], args["hidden_dim2"], args["z_dim"])
     decoder_nn = vae_decoder(args["hidden_dim2"], args["hidden_dim1"], len(args["x"]))
     rng_key, rng_key_samp, rng_key_init = random.split(args["rng_key"], 3)

     def GP(kernel, params, x, y=None):
          # params = (var, length, noise)
          # set uninformative log-normal priors on our three kernel hyperparameters

          # compute kernel
          k = kernel(x, x, params[0], params[1], params[2])

          # sample Y according to the standard gaussian process formula
          numpyro.sample(
               "y",
               dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), 
               obs=y
          )

     predictive = Predictive(GP, num_samples=args["batch_size"])
     # initialise with a sample batch
     sample_batch = predictive(rng_key_samp, 
                              kernel=args["kernel"], 
                              params=(args["var"], 0.1, args["noise"]),
                              x=args["x"])
     svi_state = svi.init(rng_key_init, sample_batch['y'])

     @jit
     def epoch_train(rng_key, svi_state, num_train):

          def body_fn(i, val):
               rng_key_i = random.fold_in(rng_key, i)
               rng_key_i, rng_key_ls = random.split(rng_key_i)

               loss_sum, svi_state = val # val -- svi_state
               # directly draw sample from the GP
               length_i = numpyro.sample("length", dist.Beta(0.2, 1.0), rng_key=rng_key_ls)
               batch = predictive(rng_key_i, 
                              params=(args["var"], length_i, args["noise"]),
                              kernel=args["kernel"], 
                              x=args["x"])
               # `update` returns (svi_state, loss)
               svi_state, loss = svi.update(svi_state, batch['y']) 
               loss_sum += loss / args['batch_size']
               return loss_sum, svi_state

          return lax.fori_loop(0, num_train, body_fn, (0.0, svi_state))
  
  # check test set error -- can also have variances?
     @jit
     def eval_test(rng_key, svi_state, num_test):

          def body_fn(i, loss_sum):
               rng_key_i = random.fold_in(rng_key, i) 
               rng_key_i, rng_key_ls = random.split(rng_key_i)
               length_i = numpyro.sample("length", dist.Beta(0.2, 1.0), rng_key=rng_key_ls)

               batch = predictive(rng_key_i, 
                                   params=(args["var"], length_i, args["noise"]),
                                   kernel=args["kernel"], 
                                   x=args["x"])
               loss = svi.evaluate(svi_state, batch['y']) / args['batch_size']
               loss_sum += loss
               return loss_sum

          loss = lax.fori_loop(0, num_test, body_fn, 0.0)
          loss = loss / num_test

          return loss

     def model(z_dim, y=None, obs_idx=None):
          # `obs_idx` is indices. Notee that Jax does not allow with boolean

          params = svi.get_params(svi_state)
          # decoder_nn = vae_decoder(args["hidden_dim1"], args["hidden_dim2"], x.shape[0])
          sigma = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))

          z = numpyro.sample("z", 
                              dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
          f = numpyro.deterministic("f", decoder_nn[1](params["decoder$params"], z))
          # f_obs = numpyro.deterministic("f_obs", jnp.ones(n_observed))

          if y is None: # durinig prediction
               numpyro.sample("y_pred", dist.Normal(f, sigma))
          else: # during inference
               numpyro.sample("y", 
                         dist.Normal(f[obs_idx], sigma), 
                         obs=y[obs_idx])


     def run_mcmc(rng_key, model, args):
          start = time.time()

          init_strategy = init_to_median(num_samples=10)
          kernel = NUTS(model, init_strategy=init_strategy)
          mcmc = MCMC(
               kernel,
               num_warmup=args["num_warmup"],
               num_samples=args["num_samples"],
               num_chains=args["num_chains"],
               thinning=args["thinning"],
               progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
          )
          mcmc.run(rng_key, args["z_dim"], args["y"], args["obs_idx"])
          mcmc.print_summary()
          print("\nMCMC elapsed time:", time.time() - start)

          return mcmc.get_samples()
          

     for i in range(args['num_epochs']):
          rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
          t_start = time.time()
          # num_train, train_idx = train_init() 
          num_train = 1000

          _, svi_state = epoch_train(rng_key_train, svi_state, num_train)

          num_test = 1000
          test_loss = eval_test(rng_key_test, svi_state, num_test)
          # infer_sin(i, rng_key_infer, args["x_test"], args["y_test"])
          print(
               "Epoch {}: loss = {} ({:.2f} s.)".format(
                    i, test_loss, time.time() - t_start
               )
          )

     rng_key, rng_key_prior, rng_key_post, rng_key_pred = random.split(rng_key, 4)

     # we may want to check how the prediction does with prior
     prior_predictive = Predictive(model, num_samples=1000)
     prior_predictions = prior_predictive(rng_key_prior, 
                                        z_dim=args["z_dim"])["y_pred"]
     mean_prior_pred = jnp.mean(prior_predictions, axis=0)
     hpdi_prior_pred = hpdi(prior_predictions, 0.9)
     print("f prior has dim ", mean_prior_pred.shape)

     # after training, we sample from the posterior 
     samples_1 = run_mcmc(rng_key_post, model, args)
     # print(samples_1.keys())
     # print(samples_1)

     # get samples from predictive distribution
     predictive = Predictive(model, samples_1)
     predictions = predictive(rng_key_pred, 
                              z_dim=args["z_dim"])["y_pred"]
     mean_post_pred = jnp.mean(predictions, axis=0)
     hpdi_post_pred = hpdi(predictions, 0.9)
     print("f posterior has dim ", mean_post_pred.shape)

     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

     # for i in range(args["num_samples"]):
     #   axs[0].plot(args["x"], prior_predictions[i], color="lightgray", alpha=0.1)
     axs[0].fill_between(args["x"], 
                         hpdi_prior_pred[0], hpdi_prior_pred[1], 
                         alpha=0.4, interpolate=True)
     axs[0].plot(args["x"], mean_prior_pred)
     axs[0].plot(args["x"], args["y"], 'o')


     # for i in range(args["num_samples"]):
     #   axs[1].plot(args["x"], predictions[i], color="lightgray", alpha=0.1)
     axs[1].plot(args["x"], mean_post_pred)
     axs[1].plot(args["x"], args["y"], 'o')
     axs[1].fill_between(args["x"], 
                         hpdi_post_pred[0], hpdi_post_pred[1], 
                         alpha=0.4, interpolate=True)
     plt.show()
     plt.close()

if __name__ == "__main__":
     def func(x):
          return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

     n = 300
     rng_key_x, rng_key_y = random.split(random.PRNGKey(0))
     # obs_idx = jnp.array([100, 150, 154, 160, 200, 210, 300, 340, 350, 355])
     obs_idx = jnp.array([0, 20, 23, 50, 54, 60, 100, 110, 117, 130, 133, 140, 170, 190])
     mask = jnp.zeros(n, dtype=bool).at[obs_idx].set(True)
     unobs_idx = jnp.arange(n)[~mask]

     # mask[unobs_idx] = True
     # print("Observation locations ", obs_idx)

     # x_ = jnp.sort(random.uniform(rng_key_x, shape=(n,)))
     x_ = jnp.arange(0, 1, 1/n)
     y_ = func(x_) + random.normal(rng_key_x, shape=(n,)) * 0.1
     y_filtered = ops.index_update(y_, unobs_idx, np.NaN)

     plt.figure()
     plt.plot(np.arange(0,1,0.001), func(np.arange(0,1,0.001)), label="y=sin(x/0.1)+noise")
     plt.scatter(x_, y_, color="lightgray", label="unobserved")
     plt.scatter(x_, y_filtered, color="orange", label="observed")
     plt.legend(loc="upper left")
     plt.show()
     plt.close()

     args = {"num_epochs": 50, # 50 does not give evident contraction around observation
        "learning_rate": 1.0e-3, 
        "batch_size": 1000, 
        "hidden_dim1": 35,
        "hidden_dim2": 30,
        "z_dim": 10,
        "x": x_,
        "y": y_filtered, 
        "n": n,
        "obs_idx": obs_idx,
        "kernel": exp_kernel,
        "var": 1,
        "noise": 0.002,
        "rng_key": random.PRNGKey(1), 
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "thinning": 3,
        }
     main(args)