Multivariate Adaptive Priors
================

## Chapter 14

<img src="../Figures/varying_effects.png" width="700" />

We’ve seen varying intercepts and varying slopes *separately*, but now
we want to consider how to create a single *joint* population model over
different parameter types.

### Corelated Varying Effects

To be able to pool across *parameter types* we need a joint adaptive
prior. The geocentric version is a multivariate normal, the maximum
entropy distribution if we only have assumptions about means, variance,
and covariances. Please note that pooling information across parameter
types is separate from the idea of having separate population models
corresponding to different types of cluster or group variables. In other
words, we could still have separate multivariate adaptive priors for
different cluster or group variables.

Simulation again comes to the rescue, helping us think through what the
model is actually doing. Here’s some of the simulated data for the cafes
example.

``` r
# Load packages.
library(tidyverse)
```

    ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ✔ ggplot2 3.4.0     ✔ purrr   0.3.4
    ✔ tibble  3.1.8     ✔ dplyr   1.0.9
    ✔ tidyr   1.2.0     ✔ stringr 1.4.0
    ✔ readr   2.1.3     ✔ forcats 0.5.1
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()

``` r
library(rethinking)
```

    Loading required package: rstan
    Loading required package: StanHeaders
    rstan (Version 2.21.8, GitRev: 2e1f913d3ca3)
    For execution on a local, multicore CPU with excess RAM we recommend calling
    options(mc.cores = parallel::detectCores()).
    To avoid recompilation of unchanged Stan programs, we recommend calling
    rstan_options(auto_write = TRUE)

    Attaching package: 'rstan'

    The following object is masked from 'package:tidyr':

        extract

    Loading required package: cmdstanr
    This is cmdstanr version 0.5.2
    - CmdStanR documentation and vignettes: mc-stan.org/cmdstanr
    - CmdStan path: /Users/marcdotson/.cmdstan/cmdstan-2.31.0
    - CmdStan version: 2.31.0
    Loading required package: parallel
    rethinking (Version 2.21)

    Attaching package: 'rethinking'

    The following object is masked from 'package:rstan':

        stan

    The following object is masked from 'package:purrr':

        map

    The following object is masked from 'package:stats':

        rstudent

``` r
alpha <- 3.5      # Population mean.
beta <- -1        # Population difference in wait times.
sigma_alpha <- 1  # Intercept standard deviation.
sigma_beta <- 0.5 # Slope standard deviation.
rho <- -0.7       # Correlation between parameter types.

# Multivariate means.
Mu <- c(alpha, beta)

# Multivariate covariance decomposed into scale and correlation.
sigmas <- c(sigma_alpha, sigma_beta)
Rho <- matrix(c(1, rho, rho, 1), nrow = 2)

# Matrix multiply to get the covariance matrix.
Sigma <- diag(sigmas) %*% Rho %*% diag(sigmas)
```

Remember when we spoke about GLMs and saw how priors you might think are
“uninformative” suddenly become very informative? This decomposition of
the covariance matrix was foreshadowed: breaking a complicated,
constrained parameter space into component pieces and then setting
priors on those pieces.

``` r
diag(sigmas) # Scale
```

         [,1] [,2]
    [1,]    1  0.0
    [2,]    0  0.5

``` r
Rho          # %*% Correlation
```

         [,1] [,2]
    [1,]  1.0 -0.7
    [2,] -0.7  1.0

``` r
Sigma        # = Covariance
```

          [,1]  [,2]
    [1,]  1.00 -0.35
    [2,] -0.35  0.25

Now we can use this multivariate normal population model to simulate
varying intercepts and slopes.

``` r
# Simulated correlated varying effects.
set.seed(42)
N_cafes <- 20
corr_effects <- MASS::mvrnorm(N_cafes, Mu, Sigma)

corr_effects
```

              [,1]       [,2]
     [1,] 2.177047 -0.3682146
     [2,] 4.279077 -0.6661827
     [3,] 3.160720 -0.8044137
     [4,] 2.722826 -1.1306971
     [5,] 2.866161 -1.4320396
     [6,] 3.658161 -0.9074616
     [7,] 2.031487 -0.3286027
     [8,] 3.810365 -0.4879623
     [9,] 1.440364 -0.3537068
    [10,] 3.640797 -0.8252318
    [11,] 2.149093 -0.6313895
    [12,] 1.144130 -0.3249977
    [13,] 4.751303 -1.8656397
    [14,] 3.851424 -0.9194315
    [15,] 3.570332 -1.2094011
    [16,] 3.079615 -0.2165336
    [17,] 3.878393 -0.8669064
    [18,] 6.240813 -1.7741778
    [19,] 6.218348 -1.2028596
    [20,] 2.185435 -0.4948436

Voila, a separate intercept and slope for each cafe produced from a
single population distribution! Note that this is similar to how we
would use the posterior to draw parameters and predict to new
clusters/groups as discussed last week where now we have a multivariate
normal with a covariance matrix since we’re pooling across parameter
types.

We can plot these varying effects jointly and look at the contours of
the bivariate normal.

``` r
# Plot correlated varying effects.
alpha_cafe <- corr_effects[,1]
beta_cafe <- corr_effects[,2]

plot(
  alpha_cafe, beta_cafe, 
  col = rangi2, 
  xlab = "Intercepts", ylab = "Slopes"
)
for (l in c(0.1, 0.3, 0.5, 0.8, 0.99)) {
  lines(
    ellipse::ellipse(Sigma, centre = Mu, level = l), 
    col = col.alpha("black", 0.2)
  )
}
```

![](../Figures/week-11-bivariate-normal-1.png)

The tilt of the ellipse is the correlation. Finally, we can simulate
data using the parameters simulated from the population model.

``` r
N_visits <- 10                              # Number of visits.
afternoon <- rep(0:1, N_visits * N_cafes/2) # Afternoon predictor.
cafe_id <- rep(1:N_cafes, each = N_visits)  # Cafe IDs.

# Linear model using the parameters for each cafe.
mu <- alpha_cafe[cafe_id] + beta_cafe[cafe_id] * afternoon

# Variation within cafes.
sigma <- 0.5

# Simulate data.
wait <- rnorm(N_visits * N_cafes, mu, sigma)
cafe_data <- list(cafe_id = cafe_id, afternoon = afternoon, wait = wait)

str(cafe_data)
```

    List of 3
     $ cafe_id  : int [1:200] 1 1 1 1 1 1 1 1 1 1 ...
     $ afternoon: int [1:200] 0 1 0 1 0 1 0 1 0 1 ...
     $ wait     : num [1:200] 2.28 1.63 2.56 1.45 1.49 ...

Now we come to the model.

Since we have decomposed the covariance matrix into scale parameters
(i.e., standard deviations) and a correlation matrix, we can set priors
on each piece separately. The `LKJcorr(2)` is a weakly informative prior
on `rho` that is skeptical of correlations at either extreme, -1 and 1.
It is a regularizing prior for correlation matrices, where a parameter
values greater than 2 is even more skeptical while `LKJcorr(1)` is a
uniform prior across correlation matrices. In general, the larger the
parameter values in the LKJ prior, the more skeptical (peaked at zero).

We also now have a single population model (i.e., the upper-level model
or adaptive regularizing prior) over *all* the random effect parameters
in the likelihood, slopes and intercepts.

``` r
# Fit the model.
fit <- ulam(
  alist(
    wait ~ normal(mu, sigma),
    mu <- alpha_cafe[cafe_id] + beta_cafe[cafe_id] * afternoon,
    c(alpha_cafe, beta_cafe)[cafe_id] ~ multi_normal(c(alpha, beta), Rho, sigma_cafe),
    alpha ~ normal(5, 2),
    beta ~ normal(-1, 0.5),
    sigma_cafe ~ exponential(1),
    sigma ~ exponential(1),
    Rho ~ lkj_corr(2)
  ), 
  data = cafe_data,
  log_lik = TRUE,
  chains = 4,
  cores = 4,
  cmdstan = TRUE
)
```

    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 3, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc
    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 4, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc
    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 23, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...

    Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 1 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 1 

    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 

    Chain 2 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 2 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 2 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 2 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 2 

    Chain 2 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 2 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 2 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 2 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 2 

    Chain 2 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 2 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 2 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 2 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 2 

    Chain 2 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 2 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 2 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 2 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 2 

    Chain 2 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 2 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 2 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 2 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 2 

    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 

    Chain 3 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 3 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 3 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 3 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 3 

    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:

    Chain 4 Exception: lkj_corr_lpdf: Correlation matrix is not positive definite. (in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/RtmpTEdjKq/model-151d245e3da27.stan', line 17, column 4 to column 24)

    Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,

    Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

    Chain 4 

    Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 1 finished in 0.8 seconds.
    Chain 2 finished in 0.9 seconds.
    Chain 4 finished in 0.8 seconds.
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.9 seconds.

    All 4 chains finished successfully.
    Mean chain execution time: 0.9 seconds.
    Total execution time: 1.1 seconds.

That’s it. Now instead of the associated scale parameter in the
population model determining the pooling across parameters of a given
type, we also have a correlation parameter in the population model
determining the pooling across parameter types.

Let’s look at the Stan code directly.

``` r
stancode(fit)
```

    data{
        vector[200] wait;
        int afternoon[200];
        int cafe_id[200];
    }
    parameters{
        vector[20] beta_cafe;
        vector[20] alpha_cafe;
        real alpha;
        real beta;
        vector<lower=0>[2] sigma_cafe;
        real<lower=0> sigma;
        corr_matrix[2] Rho;
    }
    model{
        vector[200] mu;
        Rho ~ lkj_corr( 2 );
        sigma ~ exponential( 1 );
        sigma_cafe ~ exponential( 1 );
        beta ~ normal( -1 , 0.5 );
        alpha ~ normal( 5 , 2 );
        {
        vector[2] YY[20];
        vector[2] MU;
        MU = [ alpha , beta ]';
        for ( j in 1:20 ) YY[j] = [ alpha_cafe[j] , beta_cafe[j] ]';
        YY ~ multi_normal( MU , quad_form_diag(Rho , sigma_cafe) );
        }
        for ( i in 1:200 ) {
            mu[i] = alpha_cafe[cafe_id[i]] + beta_cafe[cafe_id[i]] * afternoon[i];
        }
        wait ~ normal( mu , sigma );
    }
    generated quantities{
        vector[200] log_lik;
        vector[200] mu;
        for ( i in 1:200 ) {
            mu[i] = alpha_cafe[cafe_id[i]] + beta_cafe[cafe_id[i]] * afternoon[i];
        }
        for ( i in 1:200 ) log_lik[i] = normal_lpdf( wait[i] | mu[i] , sigma );
    }

The biggest change here is that the covariance matrix is composed of
matrix multiplication of `Rho` and `sigma_cafe`. There is a helper
function in Stan called `quad_form_diag()` that does this for us. Note
that if we wanted to get `Sigma` as a parameter with its own posterior
distribution, we would need to include this `quad_form_diag()` resulting
in `Sigma` in the `generated quantities` block.

This translation of Stan code from `ulam()` is just a starting point. We
can write better and cleaner code directly in Stan, especially as we
implement a non-centered parameterization.

### Non-Centered Parameterization Revisted

When we have correlated varying effects, we typically want a
non-centered parameterization, which is complicated by the fact that we
now have a covariance matrix decomposed into scale parameters and a
correlation matrix. However, the steps we’ve already detailed still
apply, now complicated by the fact that we have a multivariate adaptive
prior.

![](../Figures/reparameterize-02.png)

Assume that our population model is now
`Beta ~ multi_normal(beta_bar, Sigma)` where `Beta` is a matrix composed
of the separate vectors for the intercepts and slopes. The non-centered
parameterization factors out the hyperparameters from the population
model in favor of a two-step process:

1.  A standard normal population model `Delta ~ normal(0, 1)`.
2.  A linear model where `Beta` is replaced by
    `beta_bar + Delta * quad_form_diag(Omega, tau)`.

This might seem complicated, but we can simplify things further by using
the `transformed parameters` block. Not only will the
`transformed paramaters` block allow us to include deterministic
transformations of other parameters as part of the output as if they
were included in the `generated quantities` block, we can also reference
any transformed parameters in the `model` block itself. In other words,
we need to only define `Sigma` and the non-centered `Beta` once!

Here’s the model above with a `transformed parameters` block for *both*
the covariance matrix decomposition and a non-centered parameterization.

    // Data block.
    data {
      vector[200] wait;              // Vector of weight times (length hard-coded).
      int afternoon[200];            // Vector of afternoon indicators (length hard-coded).
      int cafe_id[200];              // Vector of cafe IDs (length hard-coded).
    }

    // Parameters block.
    parameters {
      matrix[20, 2] Delta;           // Non-centered varying intercepts and slopes.
      real<lower=0> sigma;           // Likelihood variance.
      matrix[1, 2] Gamma;            // Population model averages.
      vector<lower=0>[2] sigma_cafe; // Population model vector of scale hyperparameters.
      corr_matrix[2] Rho;            // Population model correlation matrix hyperparameters.
    }

    // Transformed parameters block.
    transformed parameters {
      cov_matrix[2] Sigma;           // Covariance matrix Sigma.
      matrix[20, 2] Beta;            // Centered varying intercepts and slopes.
      
      // Recompose the covariance matrix Sigma.
      Sigma = quad_form_diag(Rho, sigma_cafe);
      
      // Recompose the centered parameterization of Beta.
      for (j in 1:20) {
        Beta[j,] = Gamma + Delta[n,] * Sigma;
      }
    }

    // Model block.
    model {
      // Vector of mu for the link function.
      vector[200] mu;
      
      // Priors.
      Rho ~ lkj_corr(2);
      sigma_cafe ~ exponential(1);
      sigma ~ exponential(1);
      Gamma[,1] ~ normal(5 , 2);
      Gamma[,2] ~ normal(-1 , 0.5);
      
      // Non-centered population model.
      for (j in 1:20) {
        Delta[j,] ~ normal(0, 1);
      }
      
      // Likelihood with the link function as a for loop.
      for (i in 1:200) {
        mu[i] = Beta[cafe_id[i], 1] + Beta[cafe_id[i], 2] * afternoon[i];
      }
      wait ~ normal(mu, sigma);
    }

    // Generated quantities block.
    generated quantities {
      vector[200] log_lik; // Vector for computing the log-likelihood.
      vector[200] mu;      // Vector of mu for the link function.
      
      // Computing the log-likelihood as a for loop (note = normal_lpdf()).
      for (i in 1:200) {
        mu[i] = Beta[cafe_id[i], 1] + Beta[cafe_id[i], 2] * afternoon[i];
      }
      for (i in 1:200) {
        log_lik[i] = normal_lpdf(wait[i] | mu[i], sigma);
      }
    }

## Updated Bayesian Workflow

> “What researchers need is some unified theory of golem engineering, a
> set of principles for designing, building, and refining
> special-purpose statistical procedures.”

1.  Data Story: Theoretical estimand and causal model.

- Begin with a **conceptual story**: Where do these data come from? What
  does the theory say?
- What is the causal model? Translate the data story into a **DAG**.

2.  Modeling: Translate the data story into a statistical model.

- Translate into probability statements (i.e., a model) by identifying
  variables, both *data* and *parameters*, and how they are distributed.
- Consider which variables should be included or excluded using a DAG
  and/or domain expertise, including where to condition on other
  variables and introduce *heterogeneity*.
- The resulting model is *generative*.

3.  Estimation: Design a statistical way to produce the estimate.

- This is your code as well as the sampling procedure, now coded for HMC
  using Stan.

4.  Test.

- Simulate data using the generative model and ensure that you can
  **recover the parameter values** you used in simulation.
- Perform an appropriate **prior predictive check** to evaluate the
  model.
- If the resulting distribution of simulated data isn’t reasonable,
  iterate on your model.
- By using informative priors, tuned by using prior predictive checks,
  you get *regularization* for free.

5.  Analyze real data.

- Use MCMC to draws samples from the posterior.
- Pay attention to diagnostics, including `Rhat`, `n_eff`, and
  especially divergent transitions. Reparameterize as needed.
- Perform an appropriate **posterior predictive check** to evaluate the
  model.
- If the resulting distribution of simulated predictions isn’t
  reasonable, iterate on your model.
- Predict outcomes by propagating the entire posterior uncertainty into
  predictions.
- Predict out-of-sample predictive fit and conduct *model comparison*.
- Go full circle and return to the DAG, manipulating the intervention
  variable of interest to produce a **counterfactual plot** to consider
  the causal implications of your analysis.
