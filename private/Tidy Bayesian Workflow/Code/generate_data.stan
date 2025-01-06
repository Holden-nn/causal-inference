// Index values, observations, covariates, and hyperior values.
data {
  int<lower = 1> N;                      // Number of observations.
  int<lower = 1> K;                      // Number of groups.
  int<lower = 1> I;                      // Number of observation-level covariates.
  
  int<lower = 1, upper = K> g[N];        // Vector of group assignments.
  int brand[N];                          // Vector of brands covariates.
  vector[N] price;                       // Vector of price covariates.
  
  real gamma_mean;                       // Mean for the hyperprior on gamma.
  real<lower = 0> gamma_var;             // Variance for the hyperprior on gamma.
  real<lower = 0> tau_min;               // Minimum for the hyperprior on tau.
  real<lower = 0> tau_max;               // Maximum for the hyperprior on tau.
  real<lower = 0> sigma_min;             // Minimum for the hyperprior on tau.
  real<lower = 0> sigma_max;             // Maximum for the hyperprior on tau.
}

// Generate data according to the hierarchical regression.
generated quantities {
  matrix[K, (I - 1)] alpha;              // Matrix of observation-level brand coefficients.
  vector[K] beta;                        // Vector of observation-level price coefficients.
  real gamma;                            // Mean of the population model.
  real<lower=0> tau;                     // Variance of the population model.
  real<lower=0> sigma;                   // Variance of the observation model.
  
  vector[N] mu;                          // Declare mu for linear model.
  real intent[N];                        // Vector of observations.

  gamma = normal_rng(gamma_mean, gamma_var);
  tau = uniform_rng(tau_min, tau_max);
  sigma = uniform_rng(sigma_min, sigma_max);

  // Draw parameter values and generate data.
  for (k in 1:K) {
    for (i in 1:(I - 1)) {
      alpha[k, i] = normal_rng(gamma, tau);
    }
    beta[k] = normal_rng(gamma, tau);
  }
  for (n in 1:N) {
    mu[n] = alpha[g[n], brand[n]] + beta[g[n]] * price[n];
    intent[n] = normal_rng(mu[n], sigma);
  }
}
