Generalized Linear Models
================

## Chapter 10

### Maximum Entropy

> “The principle of maximum entropy helps us choose likelihood functions
> by providing a way to use stated assumptions about constraints on the
> outcome variable to choose the likelihood function that is the most
> conservative distribution compatabible with the known constraints.”

In other words, maximum entropy guides us into the middle ground between
knowing nothing about the data generating process or the parameters and
accidentally (or otherwise) being too informative or overstating our
knowledge about the data generating process and the parameters. Common
distributions naturally result when using maximum entropy as guide when
specifying likelihoods and priors conditioned on known constraints.

- The **normal distribution** results from a continuous outcome with a
  finite variance.
- The **binomial distribution** results from a discrete outcome with
  two, unordered levels and a constant expected value.

Many maximum entropy distributions are part of the exponential family.

<img src="../Figures/exponential_family.png" width="700" />

### Generalized Linear Models

> “By using all of our prior knowledge about the outcome variable,
> usually in the form of constraints on the possible values it can take,
> we can appeal to maximum entropy for the choice of distribution. Then
> all we have to do is generalize the linear regression strategy –
> replace a parameter describing the shape of the likelihood with a
> linear model – to probability distributions other than the Gaussian.”

Note that “prior knowledge about the outcome variable” indicates that
using a generalized linear model is a matter of modeling and not the
data. In other words, don’t look at the data to determine your
likelihood – that’s as bad as setting priors to try and match summary
statistics. Modeling is about the data generating process while the data
we have is just one realization of that process. Deciding on the
likelihood and prior using the data is just another form of
*overfitting*.

A binomial model is a natural first step into GLMs. We go from:

    y ∼ Normal(mu_i, sigma)
    mu_i = beta0 + beta1 * x1

to

    y ∼ Binomial(n, p_i)
    f(p_i) = beta0 + beta1 * x1

We know `y` is binary and appealing to maximum entropy we know a
binomial distribution is the way to model it. But what’s up with the
`f()`? `f()` is a **link function**. Why didn’t we have a link function
when we had a normal likelihood? Because of this:

<img src="../Figures/link_functions.png" width="700" />

The linear model on `mu` needed no link function because both `mu` and
`y` were the same units: continuous values without constraint on the
number line. However, not only is `y` for a binomial binary instead of
continuous, but a linear model on `p` needs help because its a
probability. Thus using a link function will map the linear model onto
the same space as the parameter we’re modeling, `p` in this case, just
like the likelihood maps the parameters onto the outcome space.

The two most common link functions are:

- The **logit link** function, which is what we need here. It maps a
  probability onto a linear space. The inverse-logit is the logistic
  function. Hence binomial regression is also called the logit model or
  logistic regression.
- The **log link** function. It maps a strictly positive value onto a
  linear space. The inverse-log is the exponential function.

GLMs open up new avenues for modeling, but they also bring problems.
Omitted variable bias is exacerbated by this mapping of the link
function and GLM. Also, it is far more difficult to interpret parameter
estimates from GLMs, hence we will be even more reliant on using
predictions to understand the implications of the model. Finally, we can
only use information criteria to compare models with the same
likelihood.

## Chapter 11

### Binomial Regression

    y ∼ Binomial(n, p_i)
    logit(p_i) = beta0 + beta1 * x1

We’ve already noted previous that the *binomial distribution* results
from a discrete outcome with two, unordered levels and a constant
expected value. We’ve also noted that the *logit link* function is
needed for the associated linear model since it maps a probability onto
a linear space.

- The inverse-logit is the logistic function. Hence the most common form
  of binomial regression is also called the **logit model** or
  **logistic regression**. Technically we could call this a Bernoulli
  regression, but Bernoulli is just a special case of the binomial where
  `n = 1`.
- There is another form of binomial regression called **aggregated
  binomial regression** where individual trials are aggregated,
  resulting in outcomes from zero to `n` rather than binary. It is the
  same model where the data are aggregated and thus modeled differently.

Let’s follow along with the binomial regression example using the
`chimpanzees` data.

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
# Load data.
data(chimpanzees)

# Create the treatment variable.
chimpanzees <- chimpanzees |> 
  as_tibble() |> 
  mutate(treatment = 1 + prosoc_left + 2 * condition)

# Confirm we have four treatment conditions.
chimpanzees |> 
  count(treatment)
```

    # A tibble: 4 × 2
      treatment     n
          <dbl> <int>
    1         1   126
    2         2   126
    3         3   126
    4         4   126

Let’s consider the model.

    pulled_left_i ∼ Binomial(1, p_i)
    logit(p_i) = alpha[actor] + beta[treatment]
    alpha ~ Normal(0, 1.5)
    beta ~ Normal(0, 0.5)

The code for the prior predictive check isn’t replicated here, but it is
important to note how essential prior predictive checks become once we
start working with GLMs. A few things to note:

- The prior predictive check here is in the *parameter space*. Again,
  depending on the model, a different prior (and thus posterior)
  predictive check will be relevant.
- In the presence of the link function and the GLM, priors you might
  think are “uninformative” suddenly become very informative. Simulate,
  simulate, simulate.
- This same sort of problem will come again whenever we work with
  constrained parameter spaces, such as setting priors for covariance
  matrices. In addition to prior predictive checks, we will find it is
  often easier to **decompose** a complicated, constrained parameter
  space into component pieces and then set priors on those pieces. It’s
  like the motivation of mean-centering and using index variables writ
  large.

Let’s fit a binomial regression using `ulam()`, including setting
`log_lik = TRUE` so we can compute model fit via WAIC and PSIS.

``` r
# Create the data list.
chimpanzees_list <- list(
  pulled_left = chimpanzees$pulled_left,
  actor = chimpanzees$actor,
  treatment = chimpanzees$treatment
)

# Fit the model.
fit_01 <- ulam(
  alist(
    pulled_left ~ dbinom(1, p),                 # pulled_left_i ∼ Binomial(1, p_i)
    logit(p) <- alpha[actor] + beta[treatment], # logit(p_i) = alpha[actor] + beta[treatment]
    alpha[actor] ~ dnorm(0, 1.5),               # alpha ~ Normal(0, 1.5)
    beta[treatment] ~ dnorm(0, 0.5)             # beta ~ Normal(0, 0.5)
  ), 
  data = chimpanzees_list,                      # Specify the data list instead of a data frame.
  log_lik = TRUE,                               # Specify that a log likelihood should be saved.
  chains = 4,                                   # Specify the number of chains.
  cores = 4,                                    # Specify the number of cores to run in parallel.
  cmdstan = TRUE                                # Specify cmdstan = TRUE to use cmdstanr instead of rstan.
)
```

    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/Rtmp7hfl2M/model-145884efc93bd.stan', line 2, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc
    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/Rtmp7hfl2M/model-145884efc93bd.stan', line 3, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc
    Warning in '/var/folders/f9/mxzvfl4s7x7c5j52qhsgms7c0000gr/T/Rtmp7hfl2M/model-145884efc93bd.stan', line 4, column 4: Declaration
        of arrays by placing brackets after a variable name is deprecated and
        will be removed in Stan 2.32.0. Instead use the array keyword before the
        type. This can be changed automatically using the auto-format flag to
        stanc

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...

    Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 1 finished in 0.9 seconds.
    Chain 2 finished in 0.8 seconds.
    Chain 3 finished in 0.8 seconds.
    Chain 4 finished in 0.8 seconds.

    All 4 chains finished successfully.
    Mean chain execution time: 0.8 seconds.
    Total execution time: 0.9 seconds.

``` r
# Print marginal posteriors.
precis(fit_01, depth = 2)
```

                   mean        sd        5.5%       94.5%     n_eff    Rhat4
    alpha[1] -0.4403478 0.3318901 -0.95688569  0.08139736  654.2800 1.001180
    alpha[2]  3.9293547 0.7305851  2.87208735  5.19044255 1062.0287 1.001133
    alpha[3] -0.7483262 0.3485991 -1.31129065 -0.19549885  781.3849 1.001930
    alpha[4] -0.7466329 0.3330013 -1.31052905 -0.20737812  700.0045 1.005277
    alpha[5] -0.4519191 0.3376797 -0.97423086  0.09557771  715.7009 1.004143
    alpha[6]  0.4796708 0.3362658 -0.05587121  1.01352345  732.0894 1.004291
    alpha[7]  1.9654622 0.4082790  1.36150025  2.63620145  809.5552 1.004803
    beta[1]  -0.0441149 0.2949510 -0.52315156  0.43386650  627.1042 1.005269
    beta[2]   0.4825043 0.2826090  0.04007979  0.92794575  616.0886 1.004270
    beta[3]  -0.3781849 0.2888177 -0.85087840  0.08898986  683.0978 1.004024
    beta[4]   0.3651679 0.2849533 -0.10202531  0.81776160  673.1747 1.004137

These are all log-odds. We can use the inverse link function
`inv_logit()` (i.e., the logistic function) to look at the estimates in
the probability space. We could also look at the differences between
treatments, known as **contrasts**, or look at predictions – the point
is once we have (samples from) the posterior, anything we compute using
the posterior also has its own posterior distribution. Remember, this is
what we called “propagating the uncertainty” from the posterior into
whatever object of inference we want.

``` r
# Extract the posterior from the fit_01.
posterior <- extract.samples(fit_01)

# Look at the posterior of contrasts.
contrast_list <- list(
  treat1_treat3 = posterior$beta[,1] - posterior$beta[,3],
  treat2_treat4 = posterior$beta[,2] - posterior$beta[,4]
)
plot(precis(contrast_list))
```

![](../Figures/week-09-chimp-differences-1.png)

As discussed in the book, the contrasts between the treatments shows
that there isn’t compelling evidence of pro-social choice in the
chimpanzees experiment.

The posterior predictive check is *involved* but only because this is
clearly the author’s area of expertise. Once again, the way we check for
reasonable priors via the prior predictive check and compare posterior
predictions against the data to check for anomalies via the posterior
predictive check is domain-dependent. Find summaries that work for your
given application.

We saw above that the parameter estimates from a binomial regression are
in log-odds. We can exponentiate the estimates to produce odds, a
**relative effect**. We can also use the inverse logit (i.e., logistic)
function, the logistic function, to look at the **absolute effect** on
the same scale as the outcome. Navigating these different ways of
looking at the same results can be tricky but is necessary.

### Other Models for Discrete Outcomes

- We can use a **poisson regression** when we don’t know *N* or we have
  a very small probability of the event occurring and a very large
  number of trials. The Poisson distribution is a special case of the
  binomial and often employs the log link function.
- We can use a **negative binomial** (i.e., **gamma-Poisson**) model
  when we want to account for heterogeneity in the expected value
  (a.k.a., the rate) of a poisson regression. It is a mixture of
  Poissons.
- If we have more than two categories in our outcome, we need a
  **multinomial logit** model or **multinomial logistic regression** in
  place of the binomial regression. This is the case for choice models.

## Chapter 12

> “This chapter is about constructing likelihood and link functions by
> piecing together the simpler components of previous chapters. Like
> legendary monsters, these hybrid likelihoods contain pieces of other
> model types. Endowed with soem properties of each piece, they help us
> model outcome variables with inconvenient, but common, properties.”

The essential concept covered in this chapter is a framework for
creating a likelihood that fits the specific realities of your given
problem. We’re really building our own golems here by taking a step away
from GLMs to construct our own models by piecing together parts of other
models. Once again, this is something of a set-up for multilevel models.

### Over-Dispersed Counts

**Over-dispersion** is the discrete analog to thicker continuous tails.
We deal with this by using **mixtures** or distributions to produce a
monstrous likelihood.

For example, take the **beta-binomial**:

    y_i ~ BetaBinomial(N_i, pbar_i, theta)
    logit(pbar_i) = alpha[index]_i
    alpha_i ~ Normal(0, 1.5)
    theta = phi + 2
    phi ~ Exponential(1)

Why would you use this? The beta-binomial assumes that each observed
count has its own probability of success. Thus we estimate a
distribution of probabilities of success instead of a single probability
of success. The monster is this new likelihood but it allows us to
account for this heterogeneity that is likely causing the
over-dispersion.

In the chapter, we see how using this model for the graduate
applications data allows for an unconfounded estimate of the causal
effect of gender even *without* accounting for the indirect causal
effect through department. Why? It is modeling unobserved heterogeneity
– in this case, an unobserved intercept for each department.

We previously noted that we can use a **negative binomial** or
**gamma-Poisson** model when we want to account for heterogeneity in the
expected value (a.k.a., the rate) of a poisson regression. Instead of a
mixture of binomials as with the beta-binomial, the gamma-Poisson is a
mixture of Poissons. As with beta-binomial, predictors are used to
change the shape of the mixture distribution rather than the expected
value of the observations.

### Zero-Inflated Outcomes

Instead of heterogeneity in probabilities or rates, what if we have
heterogeneity in *likelihoods*? A **mixture model** is essentially that.
We see them applied for **zero-inflated outcomes**, where we have a lot
fo zeroes and those zeroes might result from two or more processes
(i.e., likelihoods).

The drinking monks example is a good one. We have two likelihood as well
as a probability parameter of the outcome being associated with either
of the likelihoods. This results in two linear models and two link
functions, with potentially two totally different sets of predictors.

### Ordered Categorical Outcomes and Predictors

More discrete models – in this case the outcome variable has **ordered
categories**. Such data require special treatment, including a new link
functions, the **cumulative link** function. A few notes as you work
through the chapter:

- Note that we only need `K - 1` intercepts since we get the last
  intercept for free according to the law of total probability.
- The ordered logit is really just a categorical distribution that take
  a vector of probabilities with a length equal to `K - 1`.
- Including an ordered categorical predictor requires composing a sum
  over the previous categories with `beta * sum(delta_j[1:K])` where the
  `delta` parameters live on the **simplex**, a vector of probabilities
  that must sum to 1, and have a **Dirichlet** prior, which produces
  probabilities across a simplex.

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

- This is your code as well as the sampling procedure, `ulam()`, which
  uses MCMC.

4.  Test.

- Simulate data using the generative model and ensure that you can
  **recover the parameter values** you used in simulation.
- Perform an appropriate **prior predictive check** to evaluate the
  model by running the model with `ulam()` (with `cmdstan = TRUE` to use
  CmdStanR as well as matching `cores` and `chains` to run in parallel)
  using `extract.prior()` to get draws from the prior and using `sim()`
  to simulate data and/or `link()` to simulate the linear model `mu` (if
  regression lines are informative). The form of the prior predictive
  check is conditioned on our model and application.
- If the resulting distribution of simulated data isn’t reasonable,
  iterate on your model.
- By using informative priors, tuned by using prior predictive checks,
  you get *regularization* for free.

5.  Analyze real data.

- Use MCMC to draws samples from the posterior. Include `log_lik = TRUE`
  in `ulam()` so we can compute model fit via WAIC and PSIS.
- Pay attention to diagnostics, including `Rhat`, `n_eff`, and
  especially divergent transitions. Reparameterize as needed.
- Perform an appropriate **posterior predictive check** to evaluate the
  model by using `sim()` to predict data and/or `link()` to predict the
  linear model `mu` (if regression lines are informative), along with
  `PI()`. The form of the posterior predictive check is conditioned on
  our model and application and is likely a mirror of the prior
  predictive check. Use `postcheck()` to plot a series of posterior
  predictive checks for an `ulam()` model.
- If the resulting distribution of simulated predictions isn’t
  reasonable, iterate on your model.
- Predict outcomes using `sim()`, which propagates the entire posterior
  uncertainty into predictions.
- Predict out-of-sample predictive fit using `compare()` to get both
  PSIS and WAIC and conduct *model comparison*.
- Go full circle and return to the DAG, manipulating the intervention
  variable of interest to produce a **counterfactual plot** to consider
  the causal implications of your analysis.
