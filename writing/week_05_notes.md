Causes, Confounds, and Colliders
================

## Chapter 5

### Improving the Workflow

We need to formalize some of the tools needed to make the Bayesian
workflow *flow* better. A few things to note:

- `quap()` is our current go-to tool for estimating models and it
  parallels the mathematical expression of the joint model.
- `extract.prior()` will extract draws for our prior predictive checks
  from the specified model.
- `extract.samples()` will extract draws for our posterior predictive
  checks from the specified model.
- `link()` propagates uncertainty from our draws into the link function
  (e.g., `mu`) from the specified model.
- `sim()` propagates uncertainty from our draws into the predicted
  outcomes conditioned on the specified model.

We’ve been doing a lot of this *by hand*, which hopefully helped build
some intuition, but this set of functions will be central to our future
efforts. Eventually we’ll replace them with other tools, using packages
like {posterior}, {ggdist}, {tidybayes}, and {bayesplot}.

### Spurious Association

Let’s illustrate spurious association and the need for multiple
regression by starting with a single predictor for the `WaffleDivorce`
problem. Here’s the model with priors based on standardized explanatory
variables:

    divorce_i ∼ Normal(mu_i, sigma)
    mu_i = beta0 + beta_age * age
    beta0 ∼ Normal(0, 0.2)
    beta_age ~ Normal(0, 0.5)
    sigma ∼ Exponential(1)

We need to standardize the data so that these priors make sense.
However, we also aren’t sure about how to set priors on standardized
data. **When in doubt, simulate with a prior predictive check.**

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
data(WaffleDivorce)

# Standardize the variables.
data <- as_tibble(WaffleDivorce) |> 
  mutate(
    divorce = standardize(Divorce),
    age = standardize(MedianAgeMarriage)
  )

# Fit the model with quadratic approximation.
fit_01 <- quap(
  alist(
    divorce ~ dnorm(mu, sigma),   # divorce_i ∼ Normal(mu_i, sigma)
    mu <- beta0 + beta_age * age, # mu_i = beta0 + beta_age * age
    beta0 ~ dnorm(0, 0.2),        # beta0 ∼ Normal(0, 0.2)
    beta_age ~ dnorm(0, 0.5),     # beta_age ~ Normal(0, 0.5)
    sigma ~ dexp(1)               # sigma ∼ Exponential(1)
  ),
  data = data
)

# Extract draws for the prior predictive check.
prior_01 <- extract.prior(fit_01)

# Use link() to compute mu without having to write out the linear function.
mu <- link(fit_01, post = prior_01, data = list(age = range(data$age)))

# Base plot plus "layers".
plot(NULL, xlim = c(-2, 2), ylim = c(-2,2))
for (i in 1:50) {
  lines(c(-2, 2), mu[i,], col = col.alpha("black",0.4))
}
```

![](../Figures/week-05-divorce-prior-pd-01-1.png)

This is still *strange*. We’re plotting lines through two points created
using the `link()` function. We get something similar by just plotting
the draws themselves.

``` r
# Plot the prior predictive distribution regression lines.
prior_01 |>
  as_tibble() |> 
  ggplot() +
  geom_abline(aes(intercept = beta0, slope = beta_age), alpha = 0.10) +
  xlim(c(-2, 2)) +
  ylim(c(-2, 2))
```

![](../Figures/week-05-divorce-prior-pd-02-1.png)

These seem reasonable, given what we know about divorce rates from the
book. We’re not constraining a positive or negative outcome. But it
might still be strange to be producing lines as a prior predictive
check. Instead, we can plot the outcome directly.

``` r
# Plot the outcome data.
sim(fit_01, post = prior_01, data = list(age = range(data$age))) |> 
  as.vector() |> 
  as_tibble() |> 
  ggplot(aes(x = value)) +
  geom_histogram()
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](../Figures/week-05-divorce-prior-pd-03-1.png)

We’ve set these priors with some help. Typically we’ll need to iterate
through this process and check the resulting predictive check against
domains expertise to ensure we’re getting something “reasonable,”
whatever that happens to be for the given application and visualization.

With reasonable priors and the model fit, we can look at posterior
predictions.

``` r
# Base plot plus "layers".
plot(divorce ~ age, data = data, col = col.alpha(rangi2,0.7))
x_seq <- seq(from = -3, to = 3.2, length.out = 30)
mu <- link(fit_01, data = list(age = x_seq))
mu_mean <- apply(mu, 2, mean)
mu_pi <- apply(mu, 2, PI)
lines(x_seq, mu_mean, lwd = 2)
shade(mu_pi, x_seq)
```

![](../Figures/week-05-divorce-posterior-pd-01-1.png)

Or looking at the outcome.

``` r
# Extract draws for the posterior predictive check.
post_01 <- extract.samples(fit_01)

# Plot the posterior predictive distribution.
sim(fit_01, post = post_01) |> 
  as.vector() |> 
  as_tibble() |> 
  ggplot(aes(x = value)) +
  geom_histogram()
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](../Figures/week-05-divorce-posterior-pd-02-1.png)

We see that the median age of marriage has a negative effect on divorce
(i.e., people getting married younger is correlated with more divorce),
which is more obvious with the lines. But there has to be more to it. In
fact, this apparent effect is likely to change once we create a more
*complete* model. But we know this. It’s time to think more deeply about
including many predictors.

### DAGs

We can express our model as a **directed acyclic graph**, which can help
provide clarity in terms of causal inference by helping us think through
*direct* and *indirect* (i.e., mediation) effects.

``` r
# Create a DAG.
library(dagitty)
library(ggdag)
```


    Attaching package: 'ggdag'

    The following object is masked from 'package:stats':

        filter

``` r
dag_01 <- dagitty("dag {
  A -> D
  A -> M
  M -> D
}")

# # Daggity plot.
# coordinates(dag_01) <- list(x = c(A = 0, D = 1, M = 2), y = c(A = 0, D = 1, M = 0))
# plot(dag_01)

# ggdag.
ggdag(dag_01, layout = "circle")
```

![](../Figures/week-05-dag-01-1.png)

`A` to `D` is the **direct effect** while `A` to `M` to `D` is the
**indirect effect**. As we move into multiple regression, we can keep
this in mind. However, the model itself doesn’t encode these
relationships – this is just us adding some structure around thinking
about the model, which variables to include, and interpreting the
parameters.

To be specific, we are typically considered with one of two types of
effects:

1.  The **total causal effect**. To estimate the total causal effect, we
    need to allow for both direct and indirect effects. Given our
    example above, if we want to estimate the total causal effect of `A`
    on `D`, we *only* include `A` in the model.
2.  The **direct causal effect**. To estimate the direct causal effect,
    we need to control for indirect effects. Given our example above, if
    we want to estimate the direct causal effect of `A` on `D`, we
    include both `A` and `M`.

### Multiple Regression

Multiple regression simply expands the notation we’ve been developing.

    divorce_i ∼ Normal(mu_i, sigma)
    mu_i = beta0 + beta_age * age + beta_marriage * marriage
    beta0 ∼ Normal(0, 0.2)
    beta_age ~ Normal(0, 0.5)
    beta_marriage ~ Normal(0, 0.5)
    sigma ∼ Exponential(1)

> “If you are like most people, this is still pretty mysterious. So it
> might help to read the + symbols as “or” and then say: A State’s
> divorce rate can be a function of its marriage rate or its median age
> at marriage. The ‘or’ indicates independent associations, which may be
> purely statistical or rather causal.”

``` r
# Load the data and standardize variables.
data(WaffleDivorce)
data <- as_tibble(WaffleDivorce) |> 
  mutate(
    divorce = standardize(Divorce),
    age = standardize(MedianAgeMarriage),
    marriage = standardize(Marriage)
  )

# Fit the model with quadratic approximation.
fit_02 <- quap(
  alist(
    divorce ~ dnorm(mu, sigma),                              # divorce_i ∼ Normal(mu_i, sigma)
    mu <- beta0 + beta_age * age + beta_marriage * marriage, # mu_i = beta0 + beta_age ...
    beta0 ~ dnorm(0, 0.2),                                   # beta0 ∼ Normal(0, 0.2)
    beta_age ~ dnorm(0, 0.5),                                # beta_age ~ Normal(0, 0.5)
    beta_marriage ~ dnorm(0, 0.5),                           # beta_marriage ~ Normal(0, 0.5)
    sigma ~ dexp(1)                                          # sigma ∼ Exponential(1)
  ),
  data = data
)

# Summarize marginal posteriors.
precis(fit_02)
```

                           mean         sd       5.5%      94.5%
    beta0         -5.644206e-07 0.09707602 -0.1551468  0.1551457
    beta_age      -6.135139e-01 0.15098357 -0.8548148 -0.3722130
    beta_marriage -6.538055e-02 0.15077305 -0.3063450  0.1755839
    sigma          7.851179e-01 0.07784337  0.6607092  0.9095267

Note how we can use the probability intervals for the marginal
posteriors to conduct something like a “significance test.” The
posterior for `beta_marriage` straddles zero, so it’s not clear whether
or not this has a relationship with `divorce` once we take the effect of
`beta_age` into account. A visualization may serve us better.

``` r
# Plot the marginal posterior intervals for the two models.
plot(coeftab(fit_01, fit_02), par = c("beta_age", "beta_marriage"))
```

![](../Figures/week-05-divorce-marginals-01-1.png)

> “All of this implies there is no, or almost no, direct causal path
> from marriage rate to divorce rate. The association between marriage
> rate and divorce rate is spurious, caused by the influence of age of
> marriage on both marriage rate and divorce rate.”

How the multiple regression model achieves this simultaneously is
illustrated with residual plots. Furthermore, the importance of
inference relative to prediction when intervention is desired is
illustrated with counterfactual plots. For example, a market simulator
is also used to run counterfactuals.

### Posterior Predictive Check

> “In addition to understanding the posterior distribution of the
> parameters, it’s important to check the model’s implied predictions
> against the observed data.”

While we use prior predictive checks to evaluate the model (especially
the priors), we use posterior predictive checks to evaluate the
estimation procedure and the model, again.

``` r
# Call link without specifying new data so it uses the original data.
mu <- link(fit_02)

# Summarize samples across cases.
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)

# Simulate observations (again no new data, so it uses original data).
divorce_sim <- sim(fit_02, n = 10000)
divorce_PI <- apply(divorce_sim, 2, PI)

# Plot posterior predictions.
plot(
  mu_mean ~ data$divorce, 
  col = rangi2, 
  ylim=range(mu_PI),
  xlab = "Observed divorce", ylab = "Predicted divorce"
)
abline(a = 0, b = 1, lty = 2)
for (i in 1:nrow(data)) {
  lines(rep(data$divorce[i], 2), mu_PI[,i], col = rangi2)
}
```

![](../Figures/week-05-divorce-posterior-pd-03-1.png)

Each prediction is a posterior. The model does well for average sort of
states. It doesn’t do well for the extremes. Thinking through why could
lead us to build a better model and iterate again through this process.
Specific to this example:

> “This suggests that having a finer view on the demographic composition
> of each State, beyond just median age at marriage, would help a lot to
> refine our understanding.”

Again, the form of the posterior predictive distribution should be
related to the form of the prior predictive distribution. These line
plots are helpful for building a kind of intuition, but as we add more
parameters (i.e., dimensions) they become impractical.

### Counterfactual Plots

Counterfactual plots are to DAGs what posterior predictive distributions
are to prior predictive distributions, a way to consider the
implications of our *causal* inference conditioned on our DAG. Some
simple steps:

1.  Pick a variable to manipulate, the intervention variable.
2.  Define the range of values to set the intervention variable to.
3.  For each value of the intervention variable, and for each sample
    from the posterior, use the causal model to simulate the values of
    other variables, including the outcome.

When we manipulate the intervention variable, we break the causal
influence of other variables on the intervention variable. It’s like a
perfectly controlled experiment because we control the values of the
intervention variable. In terms of the DAG, we are deleting any of the
paths that go *into* the intervention variable.

## Chapter 6

If we only cared about prediction, we could just add everything into the
model and let it run. Additionally, we may be especially worried about
**omitted variable bias**, or not getting good estimates of the
parameters because we’ve left out some important variable (see the
Fork). However, when we care about making inferences – specifically, we
want to use the posterior distribution to inform some decision. And that
requires more thought. There has to be a reason for adding predictors,
otherwise we will run into various flavors of **included variable bias**
(see the Pipe and the Collider). This is part of [telling the
story](https://clever-kepler-fe8df7.netlify.com/).

This chapter also provides a more detailed use of the idea of
*simulating (or generating) data*. We’ve done this a bit, including with
prior and posterior predictive checks. Since our model is generative, we
can simulate data from it. In the context of this chapter, when we’d
like to demonstrate or explore certain behaviors in the data that we
wouldn’t know were present in real data, we can simply simulate data
that behaves that way. Once again, when in doubt, *simulate*.

### Multicollinearity

> “Multicollinearity means very strong correlation between two or more
> predictor variables.”

This problem should be clear from how we interpret the parameters in a
multiple regression: *The effect of one predictor, controlling for all
other predictors.* How do we control for another variable that is
essentially identical (read: very strongly correlated)? The example is a
simple illustration of simulating data:

``` r
# Set the number of individuals.
N <- 100

# Set the seed so we get the same simulated values when we re-knit.
set.seed(45)

# Simulate the data and store as data frame (tibble).
data <- tibble(height = rnorm(N, 10, 2)) |> 
  mutate(leg_prop = runif(N, 0.4, 0.5)) |> 
  mutate(
    leg_left = leg_prop * height + rnorm(N, 0, 0.02),
    leg_right = leg_prop * height + rnorm(N, 0, 0.02)
  )

data
```

    # A tibble: 100 × 4
       height leg_prop leg_left leg_right
        <dbl>    <dbl>    <dbl>     <dbl>
     1  10.7     0.403     4.32      4.31
     2   8.59    0.432     3.72      3.72
     3   9.24    0.486     4.49      4.49
     4   8.51    0.441     3.78      3.74
     5   8.20    0.487     3.99      4.00
     6   9.33    0.418     3.85      3.91
     7   9.00    0.478     4.32      4.29
     8   9.65    0.411     3.93      3.97
     9  13.6     0.426     5.84      5.80
    10   9.54    0.435     4.14      4.15
    # … with 90 more rows

Note that unlike a prior predictive check, we aren’t drawing from a
prior of possible values – we are assuming we know the truth; for
example, we set the population mean height at 10. This is our simulation
and we can do as we please. We will see later that simulating data where
we know the truth is used for more than just exploring selection bias,
it is actually very helpful for making sure the model is working as
intended.

``` r
# Fit the model using simulated data.
fit_01 <- quap(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- beta0 + betal * leg_left + betar * leg_right,
    beta0 ~ dnorm(10, 100),
    betal ~ dnorm(2, 10),
    betar ~ dnorm(2, 10),
    sigma ~ dexp(1)
  ),
  data = data
)

# Summarize the posterior.
precis(fit_01)
```

               mean         sd       5.5%     94.5%
    beta0 0.3687462 0.30318322 -0.1157991 0.8532916
    betal 0.6284772 2.35512687 -3.1354704 4.3924248
    betar 1.5344440 2.35730593 -2.2329861 5.3018742
    sigma 0.6525788 0.04592004  0.5791897 0.7259679

``` r
plot(precis(fit_01))
```

![](../Figures/week-05-sim-posterior-01-1.png)

The estimates are strange. Looking at the posterior distribution for the
two betas is illustrative.

``` r
# Draw samples from the posterior and plot.
post_01 <- extract.samples(fit_01)
plot(betal ~ betar, post_01, col = col.alpha(rangi2, 0.1), pch = 16)
```

![](../Figures/week-05-sim-posterior-02-1.png)

Correlated much?!

> “The posterior distribution for these two parameters is very highly
> correlated, with all of the plausible values of \[betal\] and
> \[betar\] lying along a narrow ridge. When \[betal\] is large, then
> \[betar\] must be small. What has happened here is that since both leg
> variables contain almost exactly the same information, if you insist
> on including both in a model, then there will be a practically
> infinite number of combinations of \[betal\] and \[betar\] that
> produce the same predictions.”

One way to think of this “infinite number of combinations” that produce
the same prediction is that these two parameters are **not identified**.
However, their sum is, as described on p. 165.

We typically need to drop one of these predictors. We’ll use the leg he
doesn’t in the book to demonstrate equivalence when choosing which
predictor to drop.

``` r
# Fit the model again using just the right leg.
fit_02 <- quap(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- beta0 + betar * leg_right,
    beta0 ~ dnorm(10, 100),
    betar ~ dnorm(2, 10),
    sigma ~ dexp(1)
  ),
  data = data
)

# Summarize the posterior.
precis(fit_02)
```

               mean         sd       5.5%     94.5%
    beta0 0.3669091 0.30318503 -0.1176391 0.8514574
    betar 2.1632650 0.06538629  2.0587650 2.2677649
    sigma 0.6527456 0.04593142  0.5793383 0.7261528

``` r
plot(precis(fit_02))
```

![](../Figures/week-05-sim-posterior-03-1.png)

Back to the sum being identified, we can compare the two marginal
posteriors.

``` r
# The sum of the two unidentified parameters are identified.
sum_blbr <- post_01$betal + post_01$betar
dens(sum_blbr, col = rangi2, lwd = 2, xlab="sum of betal and betar")
```

![](../Figures/week-05-sim-posterior-04-1.png)

``` r
# The single identified parameter from the second model looks the same.
post_02 <- extract.samples(fit_02)
dens(post_02$betar, col = rangi2, lwd = 2, xlab="betar")
```

![](../Figures/week-05-sim-posterior-04-2.png)

See how similar they are? The moral of the story:

> “When two predictor variables are very strongly correlated, including
> both in a model may lead to confusion.”

And yet again we get a highlight of the difference between modeling for
inference and prediction:

> “The posterior distribution isn’t wrong, in such cases. It’s telling
> you that the question you asked cannot be answered with these data.
> And that’s a great thing for a model to say, that it cannot answer
> your question. And if you are just interested in prediction, you’ll
> find that this leg model makes fine predictions. It just doesn’t make
> any claims about which leg is more important.”

Finally, the inevitable question:

> “How strong does a correlation have to get, before you should start
> worrying about multicollinearity? There’s no easy answer to that
> question. Correlations do have to get pretty high before this problem
> interferes with your analysis. And what matters isn’t just the
> correlation between a pair of variables. Rather, what matters is the
> correlation that remains after accounting for any other predictors. So
> really what you need, as always, is some conceptual model for how
> these variables produce observations… We always need conceptual
> models, based upon scientific background, to do useful statistics. The
> data themselves just aren’t enough.”

### The Pipe

> “\[M\]istaken inferences arising from including variables that are
> consequences of other variables.”

Treating soil to reduce fungus and improve plant health (e.g., height)
is a nice example, but very biological. Instead of plants, let’s
consider consumers where we measure their likelihood to purchase a
product. The treatment is an advertisement and the post-treatment effect
is their brand recall. If we include brand recall (the fungus, if you
follow me), we can attribute possible change in likelihood to purchase
to the brand recall rather than the treatment – the advertisement.

We can follow along with the simulation example and superimpose this new
application.

``` r
# Set the number of individuals.
N <- 100

# Set the seed so we get the same simulated values when we re-knit.
set.seed(45)

# Simulate the data and store as data frame (tibble).
data <- tibble(rating_pre = rnorm(N, 10, 2)) |> 
  mutate(treatment = rep(0:1, each = N/2)) |> 
  mutate(
    recall = rbinom(N, size = 1, prob = 0.5 - treatment * 0.4),
    rating_post = rating_pre + rnorm(N, 5 - 3 * recall)
  )

precis(data)
```

                    mean        sd      5.5%    94.5%     histogram
    rating_pre  10.16237 2.2680169  6.897643 13.89679 ▁▂▂▅▅▇▂▃▁▁▁▁▁
    treatment    0.50000 0.5025189  0.000000  1.00000    ▇▁▁▁▁▁▁▁▁▇
    recall       0.27000 0.4461960  0.000000  1.00000    ▇▁▁▁▁▁▁▁▁▂
    rating_post 14.20962 3.1203439 10.018921 19.73350     ▁▂▇▇▅▇▁▁▁

> “When designing the model, it helps to pretend you don’t have the data
> generating process just above. In real research, you will not know the
> real data generating process. But you will have a lot of scientific
> information to guide model construction.”

Remember that when we are simulating data in this way we *know* the
truth – we decide what is true – and we *never* know the truth with real
data. However, we need to be able to think clearly through the
conceptual story of our data and how we translate that into a
statistical model. The discussion in the book around how to create this
model is a helpful illustration of applying domain expertise to create a
more thoughtful model.

``` r
# Fit the model.
fit_03 <- quap(
  alist(
    rating_post ~ dnorm(mu, sigma),
    mu <- rating_pre * p,
    p <- beta0 + betat * treatment + betar * recall,
    beta0 ~ dlnorm(0, 0.2),
    betat ~ dnorm(0, 0.5),
    betar ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), 
  data = data
)

# Summarize the marginal posteriors.
precis(fit_03)
```

                 mean         sd        5.5%       94.5%
    beta0  1.43801372 0.02262175  1.40185980  1.47416764
    betat  0.01206266 0.02816300 -0.03294725  0.05707256
    betar -0.24705340 0.03336799 -0.30038189 -0.19372492
    sigma  1.24967708 0.08757678  1.10971247  1.38964169

``` r
plot(precis(fit_03))
```

![](../Figures/week-05-bias-posterior-01-1.png)

Including both the treatment and the post-treatment effect (recall)
makes it appear that the treatment has no effect and recall actually
reduces the likelihood of purchase, which we know isn’t true (we
simulated the data, remember). Let’s drop the post-treatment variable
that’s causing the problem.

``` r
# Fit the model.
fit_04 <- quap(
  alist(
    rating_post ~ dnorm(mu, sigma),
    mu <- rating_pre * p,
    p <- beta0 + betat * treatment,
    beta0 ~ dlnorm(0, 0.2),
    betat ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), 
  data = data
)

# Summarize the marginal posteriors.
precis(fit_04)
```

               mean         sd       5.5%     94.5%
    beta0 1.3257594 0.02101664 1.29217077 1.3593481
    betat 0.1222317 0.02984170 0.07453886 0.1699245
    sigma 1.5575359 0.10888461 1.38351725 1.7315545

``` r
plot(precis(fit_04))
```

![](../Figures/week-05-bias-posterior-02-1.png)

The treatment now has the positive impact we specified when simulating
the data. This illustrates what he calls **d-separation** in the book.
An equivalent (and perhaps more useful) term is **conditional
independence**. In this case, the outcome is independent of the
treatment conditioned on the post-treatment bias.

> “The problem of post-treatment variables applies just as well to
> observational studies as it does to experiments. But in experiments,
> it can be easy to tell which variables are pre-treatment, like
> \[rating_pre\], and which are post-treatment, like \[recall\]. In
> observational studies, it is harder to know. But that just makes
> having a clear causal model even more important. Just tossing
> variables into a regression model, without pausing to think about path
> relationships, is a bad idea.”

### The Collider

> “When you condition on a collider, it creates statistical – but not
> necessarily causal – associations among its causes.”

In the happiness study:

> “Happiness (H) and age (A) both cause marriage (M). Marriage is
> therefore a collider. Even though there is no causal association
> between happiness and age, if we condition on marriage – which means
> here, if we include it as a predictor in a regression – then it will
> induce a statistical association between age and happiness. And this
> can mislead us to think that happiness changes with age, when in fact
> it is constant.”

To demonstrate, we turn again to a simulation. This time we have an even
more thought-out model using agent-based theory. We can see how the
conceptual model plays out by looking at the inner workings of the
function.

``` r
sim_happiness
```

    function (seed = 1977, N_years = 1000, max_age = 65, N_births = 20, 
        aom = 18) 
    {
        set.seed(seed)
        H <- M <- A <- c()
        for (t in 1:N_years) {
            A <- A + 1
            A <- c(A, rep(1, N_births))
            H <- c(H, seq(from = -2, to = 2, length.out = N_births))
            M <- c(M, rep(0, N_births))
            for (i in 1:length(A)) {
                if (A[i] >= aom & M[i] == 0) {
                    M[i] <- rbern(1, inv_logit(H[i] - 4))
                }
            }
            deaths <- which(A > max_age)
            if (length(deaths) > 0) {
                A <- A[-deaths]
                H <- H[-deaths]
                M <- M[-deaths]
            }
        }
        d <- data.frame(age = A, married = M, happiness = H)
        return(d)
    }
    <bytecode: 0x7fcbad285858>
    <environment: namespace:rethinking>

Now, to use the function to simulate data.

``` r
# Simulate the data using the sim_happiness function.
data <- sim_happiness(seed = 1977, N_years = 1000) |>
  filter(age > 17) |>                   # Only keep adults (18 and older).
  mutate(A = (age - 18) / (65 - 18)) |> # Recode age as 0 for 18 and 1 for 65.
  mutate(mid = married + 1)              # Married status index.

precis(data)
```

                       mean         sd        5.5%      94.5%  histogram
    age        4.150000e+01 13.8606201 20.00000000 63.0000000 ▃▇▇▇▇▇▇▇▇▇
    married    4.072917e-01  0.4915861  0.00000000  1.0000000 ▇▁▁▁▁▁▁▁▁▅
    happiness -1.000070e-16  1.2145867 -1.78947368  1.7894737   ▇▅▇▅▅▇▅▇
    A          5.000000e-01  0.2949068  0.04255319  0.9574468 ▇▇▇▅▇▇▅▇▇▇
    mid        1.407292e+00  0.4915861  1.00000000  2.0000000 ▇▁▁▁▁▁▁▁▁▅

Now to compare models.

``` r
# Fit the model.
fit_05 <- quap(
  alist(
    happiness ~ dnorm(mu, sigma),
    mu <- a[mid] + bA * A,
    a[mid] ~ dnorm(0, 1),
    bA ~ dnorm(0, 2),
    sigma ~ dexp(1)
  ), 
  data = data
)

# Summarize the marginal posteriors.
precis(fit_05)
```

    2 vector or matrix parameters hidden. Use depth=2 to show them.

                mean        sd       5.5%      94.5%
    bA    -0.7490274 0.1132011 -0.9299447 -0.5681102
    sigma  0.9897080 0.0225580  0.9536559  1.0257600

``` r
plot(precis(fit_05))
```

    2 vector or matrix parameters hidden. Use depth=2 to show them.

![](../Figures/week-05-happy-posterior-01-1.png)

Age clearly has a negative effect on happiness.

``` r
# Fit the model.
fit_06 <- quap(
  alist(
    happiness ~ dnorm(mu, sigma),
    mu <- a + bA * A,
    a ~ dnorm(0, 1),
    bA ~ dnorm(0, 2),
    sigma ~ dexp(1)
  ), 
  data = data
)

# Summarize the marginal posteriors.
precis(fit_06)
```

                   mean         sd       5.5%     94.5%
    a      1.649248e-07 0.07675015 -0.1226614 0.1226617
    bA    -2.728620e-07 0.13225976 -0.2113769 0.2113764
    sigma  1.213188e+00 0.02766080  1.1689803 1.2573949

``` r
plot(precis(fit_06))
```

![](../Figures/week-05-happy-posterior-02-1.png)

And when we remove marriage status, age clearly doesn’t effect
happiness.

> “The pattern above is exactly what we should expect when we condition
> on a collider. The collider is marriage status. It a common
> consequence of age and happiness. As a result, when we condition on
> it, we induce a spurious association between the two causes.”

So just don’t include the collider, right?

> “But it isn’t always so easy to see a potential collider, because
> there may be unmeasured causes. Unmeasured causes can still induce
> collider bias. So I’m sorry to say that we also have to consider the
> possibility that our DAG may be haunted.”

### The Backdoor Criteria

> “Confounding is any context in which the association between an
> outcome Y and a predictor of interest X is not the same as it would
> be, if we had experimentally determined the values of X.”

How do we deal with it? We need to isolate the causal path of interest
so that information only flows along that path. We do that by blocking
*confounding paths*, this is known as **shutting the back door** or
using the **backdoor criteria**.

> “The metaphor in play is that we don’t want any spurious correlation
> sneaking in through the back.”

Here’s what we do once we have a DAG.

1.  List all the paths connecting `X` (the potential cause of interest)
    and `Y` (the outcome). When identifying paths we ignore the
    direction of the arrows in the DAG.
2.  Classify each path by whether it is a backdoor path. Any path
    connecting `X` and `Y` that isn’t causal is a backdoor path.
3.  Classify each path as open or closed and close any open backdoor
    paths if possible.
4.  Finally, if we are interested in the direct causal effect rather
    than the total causal effect, we need to control for indirect
    effects by closing open causal paths that aren’t the direct path of
    interest.

Here are the four elemental confounds and how we close them.

1.  Fork: `X <- Z -> Y`, the classic confounder. Condition on `Z` to
    close it.
2.  Pipe: `X -> Z -> Y` as shown with post-treatment bias. Condition on
    `X` or `Z` to close it.
3.  Collider: `X -> Z <- Y`. Already closed. Don’t condition on `Z` to
    keep it closed.
4.  Descendant: `X -> Z -> Y` where `Z -> K` and `K` is a proxy.
    Condition on `Z`, or `K` if `Z` is unobserved, to partially close
    it.

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
- The resulting model is *generative*.

3.  Estimation: Design a statistical way to produce the estimate.

- This is your code as well as the sampling procedure, `quap()`.
  (Eventually, MCMC.)

4.  Test.

- Simulate data using the generative model and ensure that you can
  **recover the parameter values** you used in simulation.
- Perform an appropriate **prior predictive check** to evaluate the
  model by running the model with `quap()`, using `extract.prior()` to
  get draws from the prior and using `sim()` to simulate data and/or
  `link()` to simulate the linear model `mu` (if regression lines are
  informative). The form of the prior predictive check is conditioned on
  our model and application.
- If the resulting distribution of simulated data isn’t reasonable,
  iterate on your model.

5.  Analyze real data.

- Use quadratic approximation to fit the model. Draw samples from the
  posterior.
- Perform an appropriate **posterior predictive checks** to evaluate the
  model using `extract.samples()` to get draws from the posterior and
  using `sim()` to predict data and/or `link()` to predict the linear
  model `mu` (if regression lines are informative), along with `PI()`.
  The form of the posterior predictive check is conditioned on our model
  and application and is likely a mirror of the prior predictive check.
- If the resulting distribution of simulated predictions isn’t
  reasonable, iterate on your model.
- Go full circle and return to the DAG, manipulating the intervention
  variable of interest to produce a **counterfactual plot** to consider
  the causal implications of your analysis.
