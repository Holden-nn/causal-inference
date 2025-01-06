Bayesian Inference
================

## Chapter 1

### Epistemology and Causality

Epistemology is the theory of knowledge. It seeks to answer the
question: “How do we know what we know?” The related field of causal
inference is an attempt to address epistemological issues in scientific
modeling. It begs us to think clearly about the *causal* model of what
we are studying before ever jumping into models or data and asks us to
bring the best of scientific theory to bear. This causal model is our
attempt to understand the **true data-generating process**.

With such a theory or causal model in mind (formally or otherwise), we
can model data, learn from the information or evidence we extract, and
update our theories about the world around us. It isn’t something that
can be reduced to null hypothesis testing in many scientific fields. And
it isn’t just true about science – it’s about knowledge generally, how
we learn, and how we can know what we know.

<img src="../Figures/theory-model-evidence.png" width="700" />

Perhaps this seminar will be more statistical *thinking* rather than
statistical *rethinking* depending on how much you’ve trained to think
in terms of null hypothesis testing and frequentist statistics, but this
paragraph summarizes the challenge and solution for statistical
education:

> “This diversity of applications helps to explain why introductory
> statistics courses are so often confusing to the initiates. Instead of
> a single method for building, refining, and critiquing statistical
> models, students are offered a zoo of pre-constructed golems known as
> ‘tests.’ … What researchers need is some unified theory of golem
> engineering, a set of principles for designing, building, and refining
> special-purpose statistical procedures.”

- Causal models are needed to learn about what is unobserved and
  unobservable.
- We should construct models based on theory so we can compare models to
  evaluate evidence for theories.
- Bayesian inference is a “single \[and self-consistent\] method for
  building, refining, and critiquing statistical models.”
- See [*The Theory That Would Not
  Die*](https://www.amazon.com/Theory-That-Would-Not-Die/dp/0300188226)

### Bayesian Statistics

> “Bayesian data analysis is no more than counting the numbers of ways
> the data could happen, according to our assumptions. Things that can
> happen more ways are more plausible. Probability theory is relevant
> because probability is just a calculus for counting… Bayesian data
> analysis is just a logical procedure for processing information.”

- Bayesian data analysis is what happens when you take probability
  theory seriously.
- Frequentist statistics is indirect because it assumes data is random
  and parameters are fixed. Bayesian statistics is direct because it
  assumes data is fixed and parameters are random.
- Randomness for a Bayesian is simply an expression of uncertainty
  rather than a result of frequency.
- We have prior beliefs about parameters before we see the data, then we
  update those beliefs with data to produce posterior beliefs which can
  be used as priors for subsequent evidence, etc.

<img src="../Figures/bayesian-triptych.png" width="700" />

The trade-off for directness is computational complexity, so let’s draw
the Bayesian owl. The beginnings of a workflow to iterate over:

1.  Theoretical estimand (data story).
2.  Scientific (causal) model.
3.  Translate into a statistical model.
4.  Simulation and recovery.
5.  Analyze real data.

### Model Comparison and Prediction

Inferential models are used to understand the underlying (and likely
unobserved) process that resulted in the data we observe. However, they
can also be used for prediction. This in contrast with what we call
“predictive models” that do only that – no inference included.

- In scientific terms, think of a model as a theory. A competing model
  is therefore a competing theory. When we compare models, we are
  finding the theory that is most likely.
- In business terms, think of a model as our paradigm of the underlying
  phenomenon (e.g., choice). Competing models describe competing
  paradigms. When we compare models, we are finding the paradigm that is
  most likely and therefore the most useful for inference and
  prediction.
- Inference can be viewed as two special forms of prediction: predicting
  the consequences of an intervention and predicting counterfactuals.

### Multilevel Models

The regression models you likely know could also be called single-level,
flat, or aggregate. However, multilevel/hierarchical models really
should be the default because, at the very least, we recognize that
heterogeneity exists. Without heterogeneity, marketing wouldn’t exist.
However, this will take some time to get to.

A specific objective of this class is to have you create a multilevel
model in Stan.

### Graphical Causal Models

> “Successful prediction does not require correct causal identification.
> In fact…predictions may actually improve when we use a model that is
> causally misleading.”

Causal identification fits squarely within the realm of inference for
this exact reason. The models run for prediction will be constructed to
ignore causality and inference in the service of prediction. We will use
DAGs a a heuristic in thinking through causality.

## Chapter 2

### Small Worlds and Large Worlds

The models (i.e., small world) represents a self-consistent theory that
may or may not be true about reality (i.e., the large world). If the
model is a good approximation of reality, it will perform well.

> “Bayesian inference is really just counting and comparing of
> possibilities.”

As a motivational example for Bayesian inference, the garden of forking
data demonstrates that a Bayesian model is a **learning model** where we
can update using *data from different sources*.

Correcting some strange coding syntax.

``` r
# ways <- c( 0 , 3 , 8 , 9 , 0 )
ways <- c(0, 3, 8, 9, 0)

# ways/sum(ways)
ways / sum(ways)
```

    [1] 0.00 0.15 0.40 0.45 0.00

- The conjectured proportion of blue marbles/surface water on Earth `p`
  is a **parameter** value.
- The conjectures about the marble/water configurations are the
  **support** of a distribution on `p`.
- The relative number of ways `p` can produce the data across
  conjectures is called the **likelihood**.
- The prior plausibility of `p` across conjectures is the **prior
  distribution**.
- The updated plausibility `p` across conjectures is the **posterior
  distribution**.

The garden of forking data is about producing the posterior
distribution. Its considering all possible parameter values (the
conjectures), specifying what we know beforehand (the prior), describing
how the data could be generated (the likelihood), and identifying the
parameter value that is most consistent with the data (the posterior).

### Bayesian Workflow

Bayesian updating is learning and the natural consequence of taking
probability seriously. Every posterior is the prior for the next
observation. The sample size is automatically embodied in the posterior
(i.e., no “degrees of freedom” or minimum sample size required before
asymptotics kick in).

A workflow is illustrated with the globe-tossing example:

    W L W W W L W L W

1.  Theoretical estimand (data story).

- Begin with a conceptual story: Where do these data come from? What
  does the theory say?

2.  Scientific (causal) model.

- What is the causal model? Translate the data story into a DAG.

3.  Translate into a statistical model.

- Translate into probability statements (i.e., a model).
- The resulting model is **generative**.

4.  Simulation and recovery.

- Simulate data using the generative model.
- Ensure that you can recover the parameter values you used in
  simulation.

5.  Analyze real data.

- Inference is conditioned on the chosen model.
- Modeling is *iterative*. You’ll likely need to revise your story.
- All models are wrong – but how are they useful?

### Model Components

> “So that we don’t have to literally count, we can use a mathematical
> function that tells us the right plausibility. In conventional
> statistics, a distribution function assigned to an observed variable
> is usually called a likelihood.”

The likelihood can also be characterized as the **observational model**.
It describes the likelihood of observing the data under different
possible values of the parameter. For example, with a binomial
likelihood, what is the likelihood of observing water 6 times out of 9
tosses if the proportion of water on Earth was 0.5?

``` r
# dbinom( 6 , size=9 , prob=0.5 )
dbinom(6, size = 9, prob = 0.5)
```

    [1] 0.1640625

The `prob` argument in this case is `p` – the parameter or proportion of
water on Earth. Parameters need an initial state, which is the prior.
Both the prior and the likelihood are expressed as probability
distributions. Combined, the prior and likelihood are the **joint
model**.

    W ~ Binomial(N, p)
    p ~ Uniform(0, 1)

### Making the Model Go

This updating – producing the posterior distribution – is the actual
challenge and its where Bayes’ theorem comes into play. We want to use
the combination of the probability of observing the data for given
parameter values with the prior probability of those given parameter
values to produce the posterior probability for given parameter values
given the data. At some level, the updating engine we use is part of the
model.

> “And while some broadly useful models like linear regression can be
> conditioned formally, this is only possible if you constrain your
> choice of prior to special forms that are easy to do mathematics with.
> We’d like to avoid forced modeling choices of this kind, instead
> favoring conditioning engines that can accommodate whichever prior is
> most useful for inference.”

This quote describes analytic solutions that require *conjugacy* between
the likelihood and prior. This means that the likelihood and prior have
to be chosen in such a way that a known posterior distribution is the
result.

Instead, we start with **grid approximation** as a useful pedagogical
device. It won’t scale as the number of parameters increases.

1.  Define the grid.
2.  Compute the value of the prior at each parameter value on the grid.
3.  Compute the likelihood at each parameter value.
4.  Compute the unstandardized posterior at each parameter value, by
    multiplying the prior by the likelihood.
5.  Finally, standardize the posterior, by dividing each value by the
    sum of all values.

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
# 1. Define the grid.
grid_appr <- tibble(grid = seq(from = 0, to = 1, length.out = 20)) |> 
  # 2. Set the prior.
  mutate(prior = rep(1, 20)) |> 
  # 3. Compute the likelihood.
  mutate(likelihood = dbinom(6, size = 9, prob = grid)) |> 
  # 4. Compute the unstandardized posterior.
  mutate(unstd_posterior = likelihood * prior) |> 
  # 5. Standardize the posterior.
  mutate(posterior = unstd_posterior / sum(unstd_posterior))

grid_appr
```

    # A tibble: 20 × 5
         grid prior likelihood unstd_posterior   posterior
        <dbl> <dbl>      <dbl>           <dbl>       <dbl>
     1 0          1 0               0          0          
     2 0.0526     1 0.00000152      0.00000152 0.000000799
     3 0.105      1 0.0000819       0.0000819  0.0000431  
     4 0.158      1 0.000777        0.000777   0.000409   
     5 0.211      1 0.00360         0.00360    0.00189    
     6 0.263      1 0.0112          0.0112     0.00587    
     7 0.316      1 0.0267          0.0267     0.0140     
     8 0.368      1 0.0529          0.0529     0.0279     
     9 0.421      1 0.0908          0.0908     0.0478     
    10 0.474      1 0.138           0.138      0.0728     
    11 0.526      1 0.190           0.190      0.0999     
    12 0.579      1 0.236           0.236      0.124      
    13 0.632      1 0.267           0.267      0.140      
    14 0.684      1 0.271           0.271      0.143      
    15 0.737      1 0.245           0.245      0.129      
    16 0.789      1 0.190           0.190      0.0999     
    17 0.842      1 0.118           0.118      0.0621     
    18 0.895      1 0.0503          0.0503     0.0265     
    19 0.947      1 0.00885         0.00885    0.00466    
    20 1          1 0               0          0          

Now to visualize the posterior.

``` r
grid_appr |> 
  ggplot(aes(x = grid, y = posterior)) +
  geom_area()
```

![](../Figures/week-03-posterior-grid-1.png)

**Quadratic approximation** scales better. We are using a normal (i.e.,
Gaussian) distribution to approximate the posterior instead of a set of
grid points.

1.  Find the posterior mode. This is usually accomplished by some
    optimization algorithm, a procedure that virtually “climbs” the
    posterior distribution, as if it were a mountain. The golem doesn’t
    know where the peak is, but it does know the slope under its feet.
    There are many well-developed optimization procedures, most of them
    more clever than simple hill climbing. But all of them try to find
    peaks.
2.  Once you find the peak of the posterior, you must estimate the
    curvature near the peak. This curvature is sufficient to compute a
    quadratic approximation of the entire posterior distribution. In
    some cases, these calculations can be done analytically, but usually
    your computer uses some numerical technique instead.

``` r
# Load package.
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
# Look at quap() documentation.
?quap

# The quap() function 1. Finds the posterior mode and 2. Estimates the curvature near the peak.
quad_appr <- quap(
  alist(
    W ~ dbinom(W + L, p), # Binomial likelihood.
    p ~ dunif(0, 1)       # Uniform prior.
  ),
  data = list(W = 6, L = 3)
)

# Results.
precis(quad_appr)
```

           mean        sd      5.5%     94.5%
    p 0.6666667 0.1571338 0.4155366 0.9177968

Eventually we’ll rely exclusively on **Markov chain Monte Carlo**.

## Chapter 3

> “The posterior distribution is a probability distribution. And like
> all probability distributions, we can imagine drawing samples from it…
> This chapter teaches you basic skills for working with samples from
> the posterior distribution… Working with samples transforms a problem
> in calculus into a problem in data summary, into a frequency format
> problem.”

### Sampling from a Grid-Approximate Posterior

``` r
# 1. Define the grid.
grid_appr <- tibble(grid = seq(from = 0, to = 1, length.out = 1000)) |> 
  # 2. Set the prior.
  mutate(prior = rep(1, 1000)) |> 
  # 3. Compute the likelihood.
  mutate(likelihood = dbinom(6, size = 9, prob = grid)) |> 
  # 4. Compute the unstandardized posterior.
  mutate(unstd_posterior = likelihood * prior) |> 
  # 5. Standardize the posterior.
  mutate(posterior = unstd_posterior / sum(unstd_posterior))

# 6. Sample from the posterior.
grid_appr_sample <- grid_appr %>% 
  slice_sample(n = 10000, weight_by = posterior, replace = TRUE)
```

Now to visualize the posterior using just the sample.

``` r
grid_appr_sample |> 
  ggplot(aes(x = grid, y = posterior)) +
  geom_area()
```

![](../Figures/week-03-posterior-grid-sample-1.png)

The sample is a collection of draws from the posterior, not the
posterior itself. However, if we have enough draws, we have an accurate
approximation of the posterior.

### Sampling to Summarize the Posterior

``` r
# Posterior probability where p < 0.5.
grid_appr_sample %>% 
  count(grid < 0.5) %>% 
  mutate(probability = n / sum(n))
```

    # A tibble: 2 × 3
      `grid < 0.5`     n probability
      <lgl>        <int>       <dbl>
    1 FALSE         8348       0.835
    2 TRUE          1652       0.165

``` r
# Visualize posterior probability where p < 0.5
grid_appr_sample |> 
  ggplot(aes(x = grid, y = posterior)) +
  geom_line() +
  geom_area(data = filter(grid_appr_sample, grid < 0.5))
```

![](../Figures/week-03-posterior-intervals-1.png)

``` r
# Posterior probability where 0.5 < p < 0.75.
grid_appr_sample %>% 
  count(grid > 0.5 & grid < 0.75) %>% 
  mutate(probability = n / sum(n))
```

    # A tibble: 2 × 3
      `grid > 0.5 & grid < 0.75`     n probability
      <lgl>                      <int>       <dbl>
    1 FALSE                       3837       0.384
    2 TRUE                        6163       0.616

``` r
# Posterior probability where 0.5 < p < 0.75.
grid_appr_sample %>% 
  ggplot(aes(x = grid, y = posterior)) +
  geom_line() +
  geom_area(data = filter(grid_appr_sample, grid > 0.5 & grid < 0.75))
```

![](../Figures/week-03-posterior-intervals-2.png)

The rethinking package provides handy functions for quickly summarizing
the posterior samples, such as `PI()`, `HPDI()`, etc. The point of all
these various intervals are to simply *summarize the posterior*, and its
simplest to do that with samples instead of doing calculus.

> “Remember, the entire posterior distribution is the Bayesian
> ‘estimate.’ It summarizes the relative plausibilities of each possible
> value of the parameter. Intervals of the distribution are just helpful
> for summarizing it. If choice of interval leads to different
> inferences, then you’d be better off just plotting the entire
> posterior distribution.”

As a bonus, there is a discussion around loss functions, which are
helpful ways to quantify costs associated with a decision (in the book,
the decision is which point estimate to use) that can be formally used
to take a posterior distribution and inform a given decision,
conditioned on the type of loss.

### Sampling to Simulate Data

Because our model is composed of probability distributions, it is
*generative*. Generative means it can generate (or *simulate*) data.
This is useful for checking our code by specifying parameter values and
*recovering* those parameter values. It is also useful for predictions,
which is just “future data,” that we can use for model evaluation.

#### Parameter Recovery

Because the model is *generative*, we can simulate data from it to test
our code by recovering parameters. When we work with real data, we never
know what the *truth* is. However, if we create our own synthetic world
where we know what the truth is, we can simulate data based on that
truth (i.e., the parameter values) and then see if our code can recover
that same value (or something close to it). This should give us
confidence that our code is working and that we can approach the problem
of working with real data.

He refers to parameter recovery, along with testing extremes in the
data, as “Test Before You Est(imate).” He creates a function to do so.

``` r
# Function to simulate tossing a globe covered p by water N times.
sim_globe <- function(p = 0.7, N = 10) {
  sample(c("W", "L"), size = N, prob=c(p, 1 - p), replace = TRUE)
}

sim_globe()
```

     [1] "W" "W" "W" "W" "W" "L" "W" "W" "W" "W"

He uses the counting procedure to demonstrate parameter recovery.

``` r
# Function to compute posterior distribution.
compute_posterior <- function(the_sample, poss = c(0, 0.25, 0.5, 0.75, 1)) {
    W <- sum(the_sample == "W") # number of W observed
    L <- sum(the_sample == "L") # number of L observed
    ways <- sapply(poss, function(q) (q*4)^W * ((1-q)*4)^L)
    post <- ways / sum(ways)
    # bars <- sapply(post, function(q) make_bar(q)) # where is make_bar() from?
    # data.frame(poss, ways, post = round(post, 3), bars)
    data.frame(poss, ways, post = round(post, 3))
}

compute_posterior(sim_globe(N = 20))
```

      poss     ways  post
    1 0.00        0 0.000
    2 0.25      243 0.000
    3 0.50  1048576 0.068
    4 0.75 14348907 0.932
    5 1.00        0 0.000

But we can also do this by using the Bayesian machinery we’re learning
about. For example, using grid approximation.

``` r
# 1. Simulate data.
sim_data <- sim_globe(p = 0.75, N = 1000)
W <- sum(sim_data == "W")
N <- length(sim_data)

# 2. Define the grid.
grid_appr <- tibble(grid = seq(from = 0, to = 1, length.out = 1000)) |> 
  # 3. Set the prior.
  mutate(prior = rep(1, 1000)) |> 
  # 4. Compute the likelihood.
  mutate(likelihood = dbinom(W, size = N, prob = grid)) |> 
  # 5. Compute the unstandardized posterior.
  mutate(unstd_posterior = likelihood * prior) |> 
  # 6. Standardize the posterior.
  mutate(posterior = unstd_posterior / sum(unstd_posterior))

# 7. Sample from the posterior.
grid_appr_sample <- grid_appr %>% 
  slice_sample(n = 10000, weight_by = posterior, replace = TRUE)

# 8. Recover parameters.
grid_appr_sample |> 
  ggplot(aes(x = grid, y = posterior)) +
  geom_area() +
  geom_vline(aes(xintercept = 0.75), color = "red")
```

![](../Figures/week-03-recover-parameters-01-1.png)

The posterior is centered close to `p = 0.75`, the truth. Parameters
recovered! Our code/estimator is working as intended. At least, it’s
working for this single simulated data set. We should consider doing
this *many* times to confirm things are *really* working as intended.

Of course, that’s for a single parameter and a binomial likelihood. We
will extend this to any likelihood for an arbitrary number of
parameters. For now, here’s a preview of the same process for a normal
likelihood with two parameters.

``` r
# Simulate data.
# 1. Specify parameter values.
beta0 <- 2
beta1 <- 7

# 2. Generate explanatory variable x and outcome variable y.
sim_data <- tibble(x = runif(100, min = 0, max = 7)) |> 
  mutate(y = rnorm(100, beta0 + beta1 * x, 1))

# Fit the model with quadratic approximation.
quad_appr <- quap(
  alist(
    # 3. Normal likelihood on y.
    y ~ dnorm(beta0 + beta1 * x, 1), 
    # 4. Normal priors on beta0 and beta1.
    beta0 ~ dnorm(0, 10),
    beta1 ~ dnorm(0, 10)
  ),
  # 5. Include data as a list.
  data = list(y = sim_data$y, x = sim_data$x)
)

# 6. Sample from the posterior.
posterior <- extract.samples(quad_appr, n = 10000)

# 7. Recover parameters.
as_tibble(posterior) |> 
  pivot_longer(
    everything(), 
    names_to = "parameters", 
    values_to = "posterior"
  ) |> 
  ggplot(aes(x = posterior, fill = parameters)) +
  geom_density(alpha = 0.5) +
  geom_vline(aes(xintercept = 2), color = "red") +
  geom_vline(aes(xintercept = 7), color = "red")
```

![](../Figures/week-03-recover-parameters-02-1.png)

All parameters recovered (for a single simulated data set)!

#### Prior and Posterior Predictive Checks

We can simulate predictions from the prior (before we have real data) to
produce a **prior predictive distribution** (i.e., a prior predictive
check) or what we would predict based on our model prior/before we have
any real data. A prior predictive check allows us to evaluate the
implications of the prior we have specified, which becomes nearly
impossible to do otherwise as we add more parameters to our model.

Note that the prior predictive distribution is not a posterior like in
the parameter recovery task. Here we will consider a distribution of the
data.

We can also simulate predictions from the posterior (i.e., forecasting
future observations) to produce a **posterior predictive distribution**
(i.e., a posterior predictive check) or what we would predict based on
our posterior. A posterior predictive check allows us to evaluate the
ways in which our joint model is failing to match reality. Thus prior
and posterior predictive checks provide helpful ways to evaluate and
iteratively adjust our model. We’ll get comfortable with both and their
place in our research workflow as we move forward.

Note that when we simulate predictions using the posterior, we use the
the posterior samples. This is what is described as *propagating
uncertainty about the parameters into uncertainty about the
predictions*. If we remember that the posterior is the parameter
estimate, this should make sense: I want to use the whole estimate when
making predictions, not some single-point summary.

Note that our posterior predictive distribution is of the same form as
the corresponding prior predictive distribution, a distribution of the
data.

## Bayesian Workflow

1.  Data Story: Theoretical estimand and causal model.

- Begin with a conceptual story: Where do these data come from? What
  does the theory say?
- What is the causal model? Translate the data story into a DAG.

2.  Modeling: Translate the data story into a generative model.

- Translate into probability statements (i.e., a model).
- The resulting model is **generative**.

3.  Estimation: Design a statistical way to produce the estimate.

- This is your code as well as the sampling procedure.
- So far it’s grid approximation or quadratic approximation. Eventually,
  MCMC.

4.  Test.

- Simulate data using the generative model.
- Ensure that you can recover the parameter values you used in
  simulation.

5.  Analyze real data.

- Inference is conditioned on the chosen model.
- Modeling is *iterative*. You’ll likely need to revise your story.
- All models are wrong – but how are they useful?
