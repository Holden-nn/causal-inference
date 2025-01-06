Linear Models and Causal Inference
================

## Chapter 4

> “Linear regression is the geocentric model of applied statistics.”

### Normal Distributions

Like the geocentric model, linear regression is stupid in the sense that
it might not accurately describe the underlying system but its useful
since it can provide helpful descriptive approximations and accurate
predictions. Why?

1.  Ontologically: Nature is full of normal distributions.
2.  Epistemologically: It just requires a mean and a variance.

All of the common statistical procedures are just linear regression.

![](../Figures/meme-regression.png)

### A Common Language

![](../Figures/modeling-language.png)

Now as we move to linear regression, the same principles of the modeling
language apply.

    y_i ∼ Normal(mu, sigma)
    mu ∼ Normal(0, 10)
    sigma ∼ Uniform(0, 10)

A few things to note:

- This model has two parameters. The modeling components and this
  language generalize no matter the number of parameters.
- Since we have two parameters, the likelihood is the evaluation of the
  combination of all possible values of the parameters, thus the
  posterior will be two-dimensional.
- We typically have separate priors for each parameter (or group of
  parameters), which says the priors are independent of each other.
- We will see grid approximation start to break down as a way to draw
  samples from the posterior.

### A Model of Height

Let’s walk through the example and practice our nascent *workflow*. Note
that in the lecture this is a model of **weight**, but the workflow and
the thought around the model still applies.

``` r
# Load packages.
library(rethinking)
```

    Loading required package: rstan

    Loading required package: StanHeaders

    Loading required package: ggplot2

    rstan (Version 2.21.8, GitRev: 2e1f913d3ca3)

    For execution on a local, multicore CPU with excess RAM we recommend calling
    options(mc.cores = parallel::detectCores()).
    To avoid recompilation of unchanged Stan programs, we recommend calling
    rstan_options(auto_write = TRUE)

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

    The following object is masked from 'package:stats':

        rstudent

``` r
library(tidyverse)
```

    ── Attaching packages
    ───────────────────────────────────────
    tidyverse 1.3.2 ──

    ✔ tibble  3.1.8     ✔ dplyr   1.0.9
    ✔ tidyr   1.2.0     ✔ stringr 1.4.0
    ✔ readr   2.1.3     ✔ forcats 0.5.1
    ✔ purrr   0.3.4     
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ tidyr::extract() masks rstan::extract()
    ✖ dplyr::filter()  masks stats::filter()
    ✖ dplyr::lag()     masks stats::lag()
    ✖ purrr::map()     masks rethinking::map()

``` r
# Load the data.
data(Howell1)

# Inspect the data.
as_tibble(Howell1)
```

    # A tibble: 544 × 4
       height weight   age  male
        <dbl>  <dbl> <dbl> <int>
     1   152.   47.8    63     1
     2   140.   36.5    63     0
     3   137.   31.9    65     0
     4   157.   53.0    41     1
     5   145.   41.3    51     0
     6   164.   63.0    35     1
     7   149.   38.2    32     0
     8   169.   55.5    27     1
     9   148.   34.9    19     0
    10   165.   54.5    54     1
    # … with 534 more rows

``` r
# Data prep.
data <- as_tibble(Howell1) |> 
  filter(age >= 18)
```

    height_i ∼ Normal(mu, sigma)
    mu ∼ Normal(178, 20)
    sigma ∼ Uniform(0, 50)

Are these priors any good? It’s time for our first **prior predictive
distribution**.

> “By simulating from this distribution, you can see what your choices
> imply about observable \[data\]. This helps you diagnose bad choices.
> Lots of conventional choices are indeed bad ones, and we’ll be able to
> see this by conducting prior predictive simulations.”

``` r
# Simulate data.
prior_pd <- tibble(
  # 1. Simulate values of mu from its prior.
  mu = rnorm(1000, 178, 20),
  # 2. Simulate values of sigma from its prior.
  sigma = runif(1000, 0, 50),
  # 3. Simulate data conditioned on the prior values.
  height = rnorm(1000, mu, sigma)
)

# Plot the prior predictive distribution.
prior_pd |> 
  ggplot(aes(x = height)) +
  geom_histogram()
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](../Figures/week-04-height-prior-pd-01-1.png)

This prior predictive distribution is the expected distribution of our
data, given how we’ve specified our likelihood and priors. Does this
look reasonable? No one has a negative height, for a start. At this
point we can iterate on how we’ve specified our likelihood and priors,
produce another prior predictive distribution and evaluate again, etc.

> “Prior predictive simulation is very useful for assigning sensible
> priors, because it can be quite hard to anticipate how priors
> influence the observable variables.”

Assuming our prior predictive distribution had produced a reasonable
expected distribution of the data, let’s go straight to computing the
posterior distribution using quadratic approximation. Note how the
syntax mirrors the model definitions we’ve already specified.

``` r
# Fit the model with quadratic approximation.
m4_1 <- quap(
  alist(
    height ~ dnorm(mu, sigma), # height_i ∼ Normal(mu, sigma)
    mu ~ dnorm(178, 20),       # mu ∼ Normal(178, 20)
    sigma ~ dunif(0, 50)       # sigma ∼ Uniform(0, 50)
  ),
  data = data
)

# Quick summary of the posterior.
precis(m4_1)
```

               mean        sd       5.5%      94.5%
    mu    154.60705 0.4119919 153.948609 155.265494
    sigma   7.73128 0.2913811   7.265597   8.196963

Remember the posterior is two-dimensional (i.e., a multidimensional
normal), but here we have estimates for each parameter separately. These
are **marginal** posteriors, the slice of the posterior that corresponds
to the given parameter.

For a single normal distribution, a mean and variance are sufficient to
describe the whole distribution. However, to characterize a
multidimensional normal distribution, we need a vector of means and a
matrix of variances and covariances.

``` r
vcov(m4_1)
```

                    mu        sigma
    mu    0.1697372983 0.0002186306
    sigma 0.0002186306 0.0849029240

This is a **variance-covariance matrix**. The diagonal are the
variances, the off-diagonals are the covariances (and its symmetric
across the diagonal). Covariances are weird to interpret, so we
typically factor this matrix into the variances and a correlation
matrix.

``` r
diag(vcov(m4_1))
```

            mu      sigma 
    0.16973730 0.08490292 

``` r
cov2cor(vcov(m4_1))
```

                   mu       sigma
    mu    1.000000000 0.001821214
    sigma 0.001821214 1.000000000

All of this is to say that to sample from the posterior we’ll need to
draw vectors of values from a multi-dimensional Normal posterior, which
will take the variance-covariance matrix into account. Let’s do that
with `extract.samples()` and summarize our posterior draws.

``` r
# Sample from the posterior.
posterior <- extract.samples(m4_1, n = 1000)

# Some possible summaries.
precis(posterior)
```

                mean        sd       5.5%      94.5%     histogram
    mu    154.608906 0.4238025 153.920531 155.308120 ▁▁▂▃▅▇▇▅▃▂▁▁▁
    sigma   7.727837 0.2811279   7.281862   8.173596    ▁▁▂▇▇▇▃▁▁▁

``` r
as_tibble(posterior) |> 
  pivot_longer(
    mu:sigma, 
    names_to = "parameters", 
    values_to = "draws"
  ) |> 
  ggplot(aes(x = draws)) +
  geom_histogram() +
  facet_wrap(~ parameters, scales = "free")
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](../Figures/week-04-height-marginal-posteriors-01-1.png)

### Linear Prediction

This kind of intercept-only model isn’t especially interesting. We want
to add covariates/independent variables/predictors – other data that
might help explain the observed heights.

> “The strategy is to make the parameter for the mean of a Gaussian
> distribution, mu, into a linear function of the predictor variable and
> other, new parameters that we invent. This strategy is often simply
> called the linear model.”

    height_i ∼ Normal(mu_i, sigma)
    mu_i = beta0 + beta1 * (weight_i - avg_weight)
    beta0 ∼ Normal(178, 20)
    beta1 ~ Normal(0, 10)
    sigma ∼ Uniform(0, 50)

A few changes from the previous model to note:

- The `i` subscript now applies to `mu` as well.
- `mu` is no longer a parameter. `mu_i` is equal to (not distributed)
  the linear combination of our parameters and predictors.
- We are centering each `weight_i` such that `beta0` is the effect of
  the average weight on height (note how if we have the average weight
  that `beta1 = 0`).
- The prior for the effect of weight on height, `beta1`, is centered
  around zero as a way to demonstrate we want to learn the sign and size
  of the effect from the data and to make it easier to set priors.

With more parameters and priors, it becomes even more important to
produce a prior predictive distribution to see what this joint model
specification implies about reasonable values of the data. **The form of
the prior predictive distribution needs to be adapted to what is most
informative for the given problem as well as what priors we are
evaluating.** For regression, it makes sense to check be composed of
regression lines when we are evaluating the priors on `beta0` and
`beta1`.

To generate height data, we’ll need to draw values of `beta0` and
`beta1` along with a range of weights to anchor the two ends, so we’ll
just use the range of the observed weights. No, this isn’t cheating,
it’s just giving us a reasonable range of weights (the predictor). We’re
not touching the observed heights (the outcome).

``` r
# 1. Specify the number of lines to simulate.
N <- 100
prior_pd <- tibble(
  # 2. Specify the line number.
  n = 1:N,
  # 3. Simulate values of beta0 and beta1 from their priors.
  beta0 = rnorm(N, 178, 20), # beta0 ∼ Normal(178, 20)
  beta1 = rnorm(N, 0, 10)    # beta1 ~ Normal(0, 10)
) |>
  # 4. Create two rows for each line, one for each end of the range.
  expand(nesting(n, beta0, beta1), weight = range(data$weight)) |> 
  mutate(
    # 5. Simulate average height using the linear model:
    # mu_i = beta0 + beta1 * (weight_i - avg_weight)
    height = beta0 + beta1 * (weight - mean(data$weight))
  )

# Plot the prior predictive distribution regression lines.
prior_pd |> 
  ggplot(aes(x = weight, y = height, group = n)) +
  geom_line(alpha = 0.10)
```

![](../Figures/week-04-height-prior-pd-02-1.png)

Note again that this particular prior predictive check has nothing to do
with `sigma`. Here we are evaluating the marginal impact of the prior on
`beta0` and `beta1` in particular. If we know anything about the
relationship between weight and height, we can see that this
distribution of the possible relationships between height and weight, as
embodied in `beta1`, are unreasonable. We need to revise the prior.
Specifically, we have every reason to believe that `beta1` should be
strictly positive.

Here’s a third attempt at the observational model and priors.

    height_i ∼ Normal(mu_i, sigma)
    mu_i = beta0 + beta1 * (weight_i - avg_weight)
    beta0 ∼ Normal(178, 20)
    beta1 ~ Log-Normal(0, 1)
    sigma ∼ Uniform(0, 50)

The change from the previous model is the strictly positive prior on
`beta1`. Once again, let’s look again at the prior predictive
distribution like we saw before.

``` r
# 1. Specify the number of lines to simulate.
N <- 100
prior_pd <- tibble(
  # 2. Specify the line number.
  n = 1:N,
  # 3. Simulate values of beta0 and beta1 from their priors.
  beta0 = rnorm(N, 178, 20), # beta0 ∼ Normal(178, 20)
  beta1 = rlnorm(N, 0, 1)    # beta1 ~ Log-Normal(0, 1)
) |>
  # 4. Create two rows for each line, one for each end of the range.
  expand(nesting(n, beta0, beta1), weight = range(data$weight)) |> 
  mutate(
    # 5. Simulate average height using the linear model:
    # mu_i = beta0 + beta1 * (weight_i - avg_weight)
    height = beta0 + beta1 * (weight - mean(data$weight)),
  )

# Plot the prior predictive distribution regression lines.
prior_pd |> 
  ggplot(aes(x = weight, y = height, group = n)) +
  geom_line(alpha = 0.10)
```

![](../Figures/week-04-height-prior-pd-03-1.png)

Much better. The effect of weight on height might be close to zero, but
it isn’t negative. Let’s go ahead and fit the model.

``` r
# Define the average weight avg_weight.
avg_weight <- mean(data$weight)

# Fit the model with quadratic approximation.
m4_2 <- quap(
  alist(
    height ~ dnorm(mu, sigma),                   # height_i ∼ Normal(mu_i, sigma)
    mu <- beta0 + beta1 * (weight - avg_weight), # mu_i = beta0 + beta1 * (x_i - avg_weight)
    beta0 ~ dnorm(178, 20),                      # beta0 ∼ Normal(178, 20)
    beta1 ~ dlnorm(0, 1),                        # beta1 ~ Log-Normal(0, 1)
    sigma ~ dunif(0, 50)                         # sigma ∼ Uniform(0, 50)
  ),
  data = data
)
```

Let’s sample from the posterior.

``` r
# Sample from the posterior.
posterior <- extract.samples(m4_2, n = 10000)
```

Likelihood, prior, fitting the model and sampling from the posterior –
we have it all! Now we just need to make sense of the posterior. We can
first look again at the *marginal* posteriors.

``` r
# Some possible summaries.
precis(posterior)
```

                 mean         sd       5.5%       94.5%   histogram
    beta0 154.6099994 0.26842606 154.179953 155.0399135 ▁▁▁▁▃▇▇▃▂▁▁
    beta1   0.9033352 0.04183533   0.836713   0.9703203    ▁▁▂▇▇▂▁▁
    sigma   5.0709110 0.19072737   4.767531   5.3717912   ▁▁▁▅▇▃▁▁▁

``` r
as_tibble(posterior) |> 
  gather(key = "parameters", value = "draws") |> 
  ggplot(aes(x = draws)) +
  geom_histogram() +
  facet_wrap(~ parameters, scales = "free")
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](../Figures/week-04-height-marginal-posteriors-02-1.png)

From `beta1`, we can say that a person who is 1 kg heavier is expected
to be, on average, 0.90 cm taller. We won’t get such an easy
interpretation looking at marginal summaries of the other parameters.

Instead of just looking at marginal summaries, we can produce a
**posterior predictive distribution**. Instead of manually constructed a
distribution of `mu` as we did for the prior predictive distribution, we
can use `link()` to generate a posterior predictive distribution of `mu`
for us that we could then plot and summarize however we’d like. While
this again is about `mu` alone – nothing about `sigma` – to construct a
*complete* posterior predictive distribution, we need to use the entire
posterior, without marginalizing out some parameter or set of
parameters.

Fortunately, there is a function for that as well. `sim()` is like
`link()`, but we can use it for simulating heights and not just the
average height `mu`. We’ll make use of both to compare the posterior
predictive distribution to the actual data.

``` r
# 1. Specify the new data we'd like to use to predict new heights.
posterior_pd <- tibble(new_weights = seq(from = 25, to = 70)) %>% 
  bind_cols(
    # 2. Simulate mu means using the posterior.
    tibble(
      mu_mean = link(m4_2, data = list(weight = .$new_weights)) |>
        apply(2, mean)
    ),
    # 3. Simulate mu intervals using the posterior.
    as_tibble(
      link(m4_2, data = list(weight = .$new_weights)) |> 
      apply(2, PI, prob = 0.99) |> 
      t()
    ),
    # 3. Simulate predicted heights using the entire posterior.
    as_tibble(
      sim(m4_2, data = list(weight = .$new_weights)) |>
      apply(2, PI, prob = 0.89) |>
      t()
    )
  )

# Plot the data along with the posterior predictive distribution.
# Note that the tidyverse equivalent is obtuse and not worth your time (see {ggdist} instead).
plot(data = data, height ~ weight)
lines(posterior_pd$new_weights, posterior_pd$mu_mean)
shade(t(cbind(posterior_pd$`1%`, posterior_pd$`100%`)), posterior_pd$new_weights)
shade(t(cbind(posterior_pd$`5%`, posterior_pd$`94%`)), posterior_pd$new_weights)
```

![](../Figures/week-04-height-posterior-pd-01-1.png)

There are a lot of ways we can visualize the posterior predictive
distribution, but ultimately we’re just comparing our predicted heights
to the heights in the data and determining, yet again, if the model is
acting in a reasonable fashion. A general rules is that **our posterior
predictive distribution should be of the same form as the corresponding
prior predictive distribution**. (The prior and posterior predictive
distributions are the first and last panels, respectively, of the
Bayesian triptych.)

Note that when we simulate from the posterior to make predictions, we
need to be sure to *propagate* our uncertainty about parameters into our
uncertainty about predictions. In other words, we need to use the entire
posterior distribution. **Using the entire posterior distribution is
what makes the analysis Bayesian.**

> “Everything that depends upon parameters has a posterior
> distribution.”

### Categories

There are a number of ways to code (i.e., encode) discrete (i.e.,
categorical) explanatory variables, with different coding strategies
suited for specific use cases. Two of the most common ways to code
discrete explanatory variables: dummy coding and index coding. Another
common approach is called effects or sum-to-one coding. For a great
walkthrough of that approach and its benefits in the context of choice
modeling, see Elea Feit’s
[post](https://eleafeit.com/posts/2021-05-23-parameterization-of-multinomial-logit-models-in-stan/).

#### Dummy Coding

Also known as indicator coding, dummy coding is the most common way to
deal with discrete variables, where a single discrete variable with `K`
levels is encoded as `K - 1` binary columns, each indicating the
presence or absence of the given level. It is an approach to identifying
the estimates of discrete explanatory levels that has a specific
contrast hard-wired.

If we include all levels of a single discrete variable, they sum up
*across columns* to a constant—to an *intercept*. If we did that with
more than one discrete variable, we would have more than one intercept
and they would no longer be identified. With dummy coding, it is typical
to include an intercept (i.e., a constant, often a column of `1`’s) and
drop the first level (i.e., the reference level) of each of the discrete
variables.

``` r
# Fit the model with quadratic approximation.
m4_3 <- quap(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- beta0 + beta1 * male + beta2 * (weight - avg_weight), 
    beta0 ~ dnorm(178, 20), # What does the intercept mean now exactly?
    beta1 ~ dnorm(0, 10),   # What is the effect of male on height?
    beta2 ~ dlnorm(0, 1),
    sigma ~ dunif(0, 50)
  ),
  data = data
)

# Sample from the posterior.
posterior <- extract.samples(m4_3, n = 10000)

# Quick summary.
precis(posterior)
```

                 mean        sd        5.5%       94.5%     histogram
    beta0 151.5592814 0.3386338 151.0223887 152.1017000 ▁▁▁▃▅▇▇▅▂▁▁▁▁
    beta1   6.4911566 0.5311835   5.6343281   7.3283045     ▁▁▁▃▇▇▃▁▁
    beta2   0.6398146 0.0412878   0.5732219   0.7062431      ▁▁▂▇▅▁▁▁
    sigma   4.2536013 0.1597787   3.9992967   4.5058175 ▁▁▁▁▃▇▇▇▃▁▁▁▁

The drawback to dummy coding in a Bayesian setting with *real data* is
that we need to specify separate priors over the contrasts rather than
the parameters themselves. This complication for setting priors is a
good reason to consider effects or index coding. Similarly, the
intercept becomes difficulty to interpret.

If you need to dummy-code quickly, `fastDummies::dummy_cols()` is a
helpful function.

``` r
# Define S=1 female and S=2 male as an integer.
data$sex <- data$male + 1

# Dummy code sex to produce a female column.
fastDummies::dummy_cols(data$sex, remove_first_dummy = TRUE)
```

        .data .data_2
    1       2       1
    2       1       0
    3       1       0
    4       2       1
    5       1       0
    6       2       1
    7       1       0
    8       2       1
    9       1       0
    10      2       1
    11      1       0
    12      2       1
    13      1       0
    14      1       0
    15      1       0
    16      2       1
    17      2       1
    18      1       0
    19      2       1
    20      1       0
    21      2       1
    22      1       0
    23      1       0
    24      2       1
    25      2       1
    26      1       0
    27      1       0
    28      1       0
    29      1       0
    30      1       0
    31      1       0
    32      2       1
    33      2       1
    34      2       1
    35      2       1
    36      1       0
    37      1       0
    38      2       1
    39      1       0
    40      2       1
    41      2       1
    42      2       1
    43      1       0
    44      1       0
    45      1       0
    46      1       0
    47      2       1
    48      1       0
    49      2       1
    50      1       0
    51      1       0
    52      1       0
    53      2       1
    54      2       1
    55      1       0
    56      2       1
    57      1       0
    58      2       1
    59      1       0
    60      2       1
    61      1       0
    62      1       0
    63      2       1
    64      2       1
    65      1       0
    66      2       1
    67      1       0
    68      1       0
    69      2       1
    70      1       0
    71      2       1
    72      2       1
    73      2       1
    74      1       0
    75      1       0
    76      1       0
    77      1       0
    78      2       1
    79      1       0
    80      1       0
    81      2       1
    82      2       1
    83      1       0
    84      2       1
    85      1       0
    86      2       1
    87      1       0
    88      2       1
    89      1       0
    90      2       1
    91      2       1
    92      2       1
    93      1       0
    94      1       0
    95      2       1
    96      2       1
    97      1       0
    98      2       1
    99      2       1
    100     1       0
    101     2       1
    102     1       0
    103     1       0
    104     2       1
    105     2       1
    106     1       0
    107     1       0
    108     1       0
    109     2       1
    110     1       0
    111     2       1
    112     1       0
    113     1       0
    114     2       1
    115     2       1
    116     1       0
    117     2       1
    118     2       1
    119     1       0
    120     1       0
    121     2       1
    122     1       0
    123     2       1
    124     1       0
    125     1       0
    126     1       0
    127     1       0
    128     2       1
    129     1       0
    130     1       0
    131     2       1
    132     2       1
    133     1       0
    134     2       1
    135     1       0
    136     2       1
    137     2       1
    138     2       1
    139     1       0
    140     2       1
    141     1       0
    142     1       0
    143     1       0
    144     1       0
    145     1       0
    146     2       1
    147     1       0
    148     1       0
    149     2       1
    150     1       0
    151     2       1
    152     1       0
    153     2       1
    154     1       0
    155     2       1
    156     1       0
    157     1       0
    158     2       1
    159     2       1
    160     1       0
    161     2       1
    162     1       0
    163     2       1
    164     2       1
    165     2       1
    166     1       0
    167     2       1
    168     1       0
    169     1       0
    170     1       0
    171     1       0
    172     2       1
    173     1       0
    174     2       1
    175     1       0
    176     2       1
    177     1       0
    178     1       0
    179     2       1
    180     1       0
    181     2       1
    182     1       0
    183     1       0
    184     1       0
    185     2       1
    186     2       1
    187     1       0
    188     2       1
    189     1       0
    190     2       1
    191     1       0
    192     2       1
    193     1       0
    194     2       1
    195     1       0
    196     1       0
    197     1       0
    198     2       1
    199     2       1
    200     1       0
    201     2       1
    202     1       0
    203     2       1
    204     1       0
    205     2       1
    206     1       0
    207     1       0
    208     1       0
    209     1       0
    210     1       0
    211     2       1
    212     2       1
    213     2       1
    214     1       0
    215     2       1
    216     2       1
    217     2       1
    218     1       0
    219     2       1
    220     1       0
    221     1       0
    222     1       0
    223     2       1
    224     1       0
    225     2       1
    226     1       0
    227     1       0
    228     2       1
    229     1       0
    230     2       1
    231     2       1
    232     1       0
    233     1       0
    234     2       1
    235     2       1
    236     1       0
    237     2       1
    238     1       0
    239     1       0
    240     1       0
    241     1       0
    242     2       1
    243     2       1
    244     1       0
    245     2       1
    246     1       0
    247     2       1
    248     1       0
    249     2       1
    250     2       1
    251     2       1
    252     2       1
    253     1       0
    254     1       0
    255     1       0
    256     1       0
    257     2       1
    258     1       0
    259     2       1
    260     2       1
    261     2       1
    262     1       0
    263     1       0
    264     1       0
    265     2       1
    266     1       0
    267     2       1
    268     1       0
    269     1       0
    270     2       1
    271     2       1
    272     1       0
    273     2       1
    274     2       1
    275     1       0
    276     2       1
    277     1       0
    278     1       0
    279     2       1
    280     1       0
    281     1       0
    282     2       1
    283     1       0
    284     2       1
    285     1       0
    286     2       1
    287     2       1
    288     2       1
    289     1       0
    290     2       1
    291     2       1
    292     2       1
    293     2       1
    294     1       0
    295     2       1
    296     1       0
    297     1       0
    298     1       0
    299     2       1
    300     2       1
    301     1       0
    302     2       1
    303     1       0
    304     2       1
    305     2       1
    306     2       1
    307     2       1
    308     1       0
    309     1       0
    310     1       0
    311     1       0
    312     1       0
    313     2       1
    314     1       0
    315     2       1
    316     1       0
    317     2       1
    318     1       0
    319     2       1
    320     1       0
    321     2       1
    322     1       0
    323     2       1
    324     2       1
    325     1       0
    326     1       0
    327     1       0
    328     2       1
    329     2       1
    330     2       1
    331     2       1
    332     2       1
    333     1       0
    334     2       1
    335     1       0
    336     2       1
    337     1       0
    338     1       0
    339     1       0
    340     1       0
    341     1       0
    342     2       1
    343     1       0
    344     2       1
    345     1       0
    346     2       1
    347     1       0
    348     1       0
    349     1       0
    350     2       1
    351     1       0
    352     2       1

#### Index Coding

Also known as one-hot encoding, index coding similarly turns each level
of a discrete variable into its own binary column. However, with index
coding we *don’t* include an intercept and *don’t* include any reference
levels. Additionally, we don’t have to provide separate binary columns
and can simply provide a single vector with the levels of the discrete
variable indexed and can set a prior over the entire set of indexed
parameters directly.

By not including reference levels, the intercept is implied by the fact
that the *implied* columns sum to a constant, as discussed previously.
But when we have more than one discrete variable we also have more than
one implied intercept. This would create an identification problem in a
frequentist setting, but in a Bayesian analysis we simply rely on the
prior to enable identification of each of the parameters. As a bonus,
the contrasts are always identified even if their constituent parameter
estimates are not.

``` r
# Fit the model with quadratic approximation.
m4_4 <- quap(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- alpha[sex] + beta[sex] * (weight - avg_weight),
    alpha[sex] ~ dnorm(178, 20),
    beta[sex] ~ dlnorm(0, 1),
    sigma ~ dunif(0, 50)
  ),
  data = data
)

# Sample from the posterior.
posterior <- extract.samples(m4_4, n = 10000)

# Quick summary.
precis(posterior, depth = 2)
```

                    mean         sd        5.5%       94.5%     histogram
    sigma      4.2407865 0.16107569   3.9814799   4.4955800 ▁▁▁▂▃▇▇▅▃▁▁▁▁
    alpha[1] 151.3772260 0.36213858 150.7959749 151.9512055        ▁▂▇▅▁▁
    alpha[2] 157.8593975 0.38824180 157.2399672 158.4838348        ▁▂▇▅▁▁
    beta[1]    0.5839439 0.05763227   0.4917845   0.6773602    ▁▁▁▅▇▅▂▁▁▁
    beta[2]    0.6955542 0.05886932   0.6029470   0.7919789    ▁▁▁▃▇▇▃▁▁▁

Index coding like this is only possible in a Bayesian framework where
the use of priors obviates the necessity of strict identification
strategies. However, it’s not without costs. We will need to specify
informative priors, which we should be doing anyway, and the model
itself *may* take longer to converge.

If you need to produce index-coded columns instead of a single vector,
`fastDummies::dummy_cols()` remains helpful.

``` r
# Index code sex as separate columns.
fastDummies::dummy_cols(data$sex, remove_first_dummy = FALSE)
```

        .data .data_1 .data_2
    1       2       0       1
    2       1       1       0
    3       1       1       0
    4       2       0       1
    5       1       1       0
    6       2       0       1
    7       1       1       0
    8       2       0       1
    9       1       1       0
    10      2       0       1
    11      1       1       0
    12      2       0       1
    13      1       1       0
    14      1       1       0
    15      1       1       0
    16      2       0       1
    17      2       0       1
    18      1       1       0
    19      2       0       1
    20      1       1       0
    21      2       0       1
    22      1       1       0
    23      1       1       0
    24      2       0       1
    25      2       0       1
    26      1       1       0
    27      1       1       0
    28      1       1       0
    29      1       1       0
    30      1       1       0
    31      1       1       0
    32      2       0       1
    33      2       0       1
    34      2       0       1
    35      2       0       1
    36      1       1       0
    37      1       1       0
    38      2       0       1
    39      1       1       0
    40      2       0       1
    41      2       0       1
    42      2       0       1
    43      1       1       0
    44      1       1       0
    45      1       1       0
    46      1       1       0
    47      2       0       1
    48      1       1       0
    49      2       0       1
    50      1       1       0
    51      1       1       0
    52      1       1       0
    53      2       0       1
    54      2       0       1
    55      1       1       0
    56      2       0       1
    57      1       1       0
    58      2       0       1
    59      1       1       0
    60      2       0       1
    61      1       1       0
    62      1       1       0
    63      2       0       1
    64      2       0       1
    65      1       1       0
    66      2       0       1
    67      1       1       0
    68      1       1       0
    69      2       0       1
    70      1       1       0
    71      2       0       1
    72      2       0       1
    73      2       0       1
    74      1       1       0
    75      1       1       0
    76      1       1       0
    77      1       1       0
    78      2       0       1
    79      1       1       0
    80      1       1       0
    81      2       0       1
    82      2       0       1
    83      1       1       0
    84      2       0       1
    85      1       1       0
    86      2       0       1
    87      1       1       0
    88      2       0       1
    89      1       1       0
    90      2       0       1
    91      2       0       1
    92      2       0       1
    93      1       1       0
    94      1       1       0
    95      2       0       1
    96      2       0       1
    97      1       1       0
    98      2       0       1
    99      2       0       1
    100     1       1       0
    101     2       0       1
    102     1       1       0
    103     1       1       0
    104     2       0       1
    105     2       0       1
    106     1       1       0
    107     1       1       0
    108     1       1       0
    109     2       0       1
    110     1       1       0
    111     2       0       1
    112     1       1       0
    113     1       1       0
    114     2       0       1
    115     2       0       1
    116     1       1       0
    117     2       0       1
    118     2       0       1
    119     1       1       0
    120     1       1       0
    121     2       0       1
    122     1       1       0
    123     2       0       1
    124     1       1       0
    125     1       1       0
    126     1       1       0
    127     1       1       0
    128     2       0       1
    129     1       1       0
    130     1       1       0
    131     2       0       1
    132     2       0       1
    133     1       1       0
    134     2       0       1
    135     1       1       0
    136     2       0       1
    137     2       0       1
    138     2       0       1
    139     1       1       0
    140     2       0       1
    141     1       1       0
    142     1       1       0
    143     1       1       0
    144     1       1       0
    145     1       1       0
    146     2       0       1
    147     1       1       0
    148     1       1       0
    149     2       0       1
    150     1       1       0
    151     2       0       1
    152     1       1       0
    153     2       0       1
    154     1       1       0
    155     2       0       1
    156     1       1       0
    157     1       1       0
    158     2       0       1
    159     2       0       1
    160     1       1       0
    161     2       0       1
    162     1       1       0
    163     2       0       1
    164     2       0       1
    165     2       0       1
    166     1       1       0
    167     2       0       1
    168     1       1       0
    169     1       1       0
    170     1       1       0
    171     1       1       0
    172     2       0       1
    173     1       1       0
    174     2       0       1
    175     1       1       0
    176     2       0       1
    177     1       1       0
    178     1       1       0
    179     2       0       1
    180     1       1       0
    181     2       0       1
    182     1       1       0
    183     1       1       0
    184     1       1       0
    185     2       0       1
    186     2       0       1
    187     1       1       0
    188     2       0       1
    189     1       1       0
    190     2       0       1
    191     1       1       0
    192     2       0       1
    193     1       1       0
    194     2       0       1
    195     1       1       0
    196     1       1       0
    197     1       1       0
    198     2       0       1
    199     2       0       1
    200     1       1       0
    201     2       0       1
    202     1       1       0
    203     2       0       1
    204     1       1       0
    205     2       0       1
    206     1       1       0
    207     1       1       0
    208     1       1       0
    209     1       1       0
    210     1       1       0
    211     2       0       1
    212     2       0       1
    213     2       0       1
    214     1       1       0
    215     2       0       1
    216     2       0       1
    217     2       0       1
    218     1       1       0
    219     2       0       1
    220     1       1       0
    221     1       1       0
    222     1       1       0
    223     2       0       1
    224     1       1       0
    225     2       0       1
    226     1       1       0
    227     1       1       0
    228     2       0       1
    229     1       1       0
    230     2       0       1
    231     2       0       1
    232     1       1       0
    233     1       1       0
    234     2       0       1
    235     2       0       1
    236     1       1       0
    237     2       0       1
    238     1       1       0
    239     1       1       0
    240     1       1       0
    241     1       1       0
    242     2       0       1
    243     2       0       1
    244     1       1       0
    245     2       0       1
    246     1       1       0
    247     2       0       1
    248     1       1       0
    249     2       0       1
    250     2       0       1
    251     2       0       1
    252     2       0       1
    253     1       1       0
    254     1       1       0
    255     1       1       0
    256     1       1       0
    257     2       0       1
    258     1       1       0
    259     2       0       1
    260     2       0       1
    261     2       0       1
    262     1       1       0
    263     1       1       0
    264     1       1       0
    265     2       0       1
    266     1       1       0
    267     2       0       1
    268     1       1       0
    269     1       1       0
    270     2       0       1
    271     2       0       1
    272     1       1       0
    273     2       0       1
    274     2       0       1
    275     1       1       0
    276     2       0       1
    277     1       1       0
    278     1       1       0
    279     2       0       1
    280     1       1       0
    281     1       1       0
    282     2       0       1
    283     1       1       0
    284     2       0       1
    285     1       1       0
    286     2       0       1
    287     2       0       1
    288     2       0       1
    289     1       1       0
    290     2       0       1
    291     2       0       1
    292     2       0       1
    293     2       0       1
    294     1       1       0
    295     2       0       1
    296     1       1       0
    297     1       1       0
    298     1       1       0
    299     2       0       1
    300     2       0       1
    301     1       1       0
    302     2       0       1
    303     1       1       0
    304     2       0       1
    305     2       0       1
    306     2       0       1
    307     2       0       1
    308     1       1       0
    309     1       1       0
    310     1       1       0
    311     1       1       0
    312     1       1       0
    313     2       0       1
    314     1       1       0
    315     2       0       1
    316     1       1       0
    317     2       0       1
    318     1       1       0
    319     2       0       1
    320     1       1       0
    321     2       0       1
    322     1       1       0
    323     2       0       1
    324     2       0       1
    325     1       1       0
    326     1       1       0
    327     1       1       0
    328     2       0       1
    329     2       0       1
    330     2       0       1
    331     2       0       1
    332     2       0       1
    333     1       1       0
    334     2       0       1
    335     1       1       0
    336     2       0       1
    337     1       1       0
    338     1       1       0
    339     1       1       0
    340     1       1       0
    341     1       1       0
    342     2       0       1
    343     1       1       0
    344     2       0       1
    345     1       1       0
    346     2       0       1
    347     1       1       0
    348     1       1       0
    349     1       1       0
    350     2       0       1
    351     1       1       0
    352     2       0       1

### Matrix Multiplication

As you start adding many explanatory variables, it may be easier to use
matrix multiplication instead of listing each variable and parameter
separately in the linear model.

``` r
# Pay attention to the dimensions of the matrix.
data_mat <- data %>% 
  select(sex, weight) |> 
  as.matrix()

str(data_mat)
```

     num [1:352, 1:2] 2 1 1 2 1 2 1 2 1 2 ...
     - attr(*, "dimnames")=List of 2
      ..$ : NULL
      ..$ : chr [1:2] "sex" "weight"

``` r
# And the dimensions of the betas (as an illustration).
beta_vec <- c(2, 3) |> as.matrix()

str(beta_vec)
```

     num [1:2, 1] 2 3

``` r
# The order of matrix multiplication matters, with dimensions that line up.
Xbeta <- data_mat %*% beta_vec

str(Xbeta)
```

     num [1:352, 1] 147.5 111.5 97.6 163.1 125.8 ...

### Curves From Lines

Linear models are additive, but not necessarily *lines*. Allowing for
curvature might be helpful. The form of the model (especially the
likelihood) depends on the data. However, as we add complexity to the
model, issues begin to emerge with overfitting and interpretation
(glimpse of the black-box nature of predictive models).

> “Better would be to have a more mechanistic model of the data, one
> that builds the non-linear relationship up from a principled
> beginning.”

**Polynomial regression** is adding squared (and higher) transformations
of predictors to the model.

**Splines** are smooth functions built out of smaller functions. They
are a possible improvement over polynomial regression in that they are
used to avoid wild swings and can modify local regions of the curve.

- Used for de-trending: Finding an “average” curve so we can consider
  micro-deviations.
- Often more uncertainty at the parameter level than the prediction
  level.

## Updated Bayesian Workflow

1.  Data Story: Theoretical estimand and causal model.

- Begin with a conceptual story: Where do these data come from? What
  does the theory say?
- What is the causal model? Translate the data story into a DAG.

2.  Modeling: Translate the data story into a statistical model.

- Translate into probability statements (i.e., a model) by identifying
  variables, both *data* and *parameters*, and how they are distributed.
- The resulting model is **generative**.

3.  Estimation: Design a statistical way to produce the estimate.

- This is your code as well as the sampling procedure.
- So far it’s grid approximation or quadratic approximation. Eventually,
  MCMC.

4.  Test.

- Simulate data using the generative model and ensure that you can
  recover the parameter values you used in simulation.
- Perform an appropriate **prior predictive check** to evaluate the
  model. If the resulting distribution of simulated data isn’t
  reasonable, iterate on your model.

5.  Analyze real data.

- Use quadratic approximation to fit the model. Draw samples from the
  posterior.
- Perform an appropriate **posterior predictive checks** to evaluate the
  model. If the resulting distribution of simulated predictions isn’t
  reasonable, iterate on your model.
