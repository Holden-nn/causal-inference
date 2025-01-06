// Index values and observations.
data {
  int<lower = 1> N_train;                  // Number of training data observations.
  int<lower = 1> N_valid;                  // Number of validation data observations.
  int<lower = 1> N_test;                   // Number of test data observations.
  int<lower = 1> I;                        // Number of covariates.
  
  int shot_made_flag[N_train];             // Vector of observations.
  matrix[N_train, I] X_train;              // Matrix of training data covariates.
  matrix[N_valid, I] X_valid;              // Matrix of validation data covariates.
  matrix[N_test, I] X_test;                // Matrix of test data covariates.
}

// Parameters.
parameters {
  vector[I] beta;                         // Vector of coefficients.
}

// Simple regression.
model {
  // LHS of logit link function.
  real p[N_train];
  
  // Priors.
  beta ~ normal(0, 1);

  // Likelihood.
  for (n in 1:N_train) {
    p[n] = inv_logit(X_train[n,] * beta);
  }
  shot_made_flag ~ binomial(1 , p);
}

// Generate predictions using the posterior.
generated quantities {
  real p_valid[N_valid];                 // LHS of logit link function for validation data.
  real y_valid[N_valid];                 // Vector of predicted observations for validation data.
  real p_test[N_test];                   // LHS of logit link function for test data.
  real y_test[N_test];                   // Vector of predicted observations for test data.

  // Generate posterior prediction distribution.
  for (n in 1:N_valid) {
    p_valid[n] = inv_logit(X_valid[n,] * beta);
    y_valid[n] = binomial_rng(1, p_valid[n]);
  }
  for (n in 1:N_test) {
    p_test[n] = inv_logit(X_test[n,] * beta);
    y_test[n] = binomial_rng(1, p_test[n]);
  }
}

