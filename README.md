# The Repo
This repo contains code for training and doing inference with VAE. 

`src/model` contains implementation of GP (`gp.py`), VAE (`vae.py`) and MCMC (`inference.py`) based on `Numpyro`. `src/test/toy_examples.py` hosts toy 1D examples including, cubic function, trignometric function and draws from GP. `src/test/test_model.py` hosts fucntions for testing each module in `src/model`. 

To run the `src/test/toy_examples.py` in the command line, type in e.g. `python src/test/toy_examples.py --num-epochs 50 --noise 0.01`.

# Models
## GP
Exponential kernel for spatial dimension 1 and 2 is implemented, as well as aggregated kernel (as a wrapper of base kernels). For denser grid, larger noise is required for numerical stability.

## VAE
Current implementation only admits symmetric encoder and decoder with 2 hidden layers. The training involves sampling a minibatch from GP (MVN for finite number of spatial locations) with a randomly generated lengthscale from $Beta(0.1, 2.0)$ (prior subject to changes) on the fly. Hence, each minibatch contains function draws from one GP. The VAE is trained with multiple epochs contianing multiple batches. For toy examples, 50 epochs generally suffices.

## Inference
Use trained decoder as a non-linear push-forawrd function of the prior (std MVN) for latent random variable $z$. Then, MCMC is used to do inference on regression $y = f(z) + \epsilon$ for unobserved $y$. Note that, since $f(z)$ is not a stochastic process, we may only do inference on spatial locations specified in the training samples of VAE (i.e. those of samples from GP).

# TODO
Several extensions to be made
- [] GP: 
     - [x] Better implementation of base kernel for both dim 1 and dim 2
     - [] Extend current working examples to aggregated kernel
     - [] Write a class for kernel?
- [] VAE:
     - [] Instead of using `stax`, try other alternatives for looping over hidden layers
     - [] Experiment with other architecture (curernt works well for simple smooth functions)
- [] Inference:
     - [] How to better facilitate distinguishing 'lengthscale' (roughly the same idea as GP, but we do not have an explicit variable to model it) and noise in the data? 
- [] Benchmark:
     - [] Add SVGP as benchmark for posterior prediction. 
     - []Try Neural process as well?
