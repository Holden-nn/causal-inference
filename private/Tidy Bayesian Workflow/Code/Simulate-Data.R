# Load packages.
library(tidyverse)

N <- 700     # Number of observations.
K <- 3       # Number of groups.
mu <- 0      # Mean of the population model.
tau <- 0.1   # Variance of the population model.
set.seed(42) # Set simulation seed.

# Simulate covariates.
X <- tibble(
  brand = as.integer(round(runif(N, 1, 5))),
  price = round(runif(N, 10, 30), 2)
)

X_temp <- X %>% 
  mutate(id = 1:N) %>% 
  spread(key = brand, value = brand, fill = 0) %>%
  mutate(
    brand1 = ifelse(`1` != 0, 1, 0),
    brand2 = ifelse(`2` != 0, 1, 0),
    brand3 = ifelse(`3` != 0, 1, 0),
    brand4 = ifelse(`4` != 0, 1, 0),
    brand5 = ifelse(`5` != 0, 1, 0)
  ) %>%
  select(brand1:brand5, price)

# Simulate betas, sigma, g, and y.
Beta <- matrix(NA, nrow = K, ncol = ncol(X_temp))
for (k in 1:K) {
  Beta[k,] <- rnorm(ncol(X_temp), mu, tau)
}
sigma <- rexp(1)
g <- sample(1:K, N, replace = TRUE)
intent <- c(NA, N)
for (n in 1:N) {
  intent[n] <- rnorm(1, Beta[g[n],] %*% t(X_temp[n,]), sigma) + 3
  if (intent[n] > 10) intent[n] <- 10
  if (intent[n] < 1) intent[n] <- 1
}

tibble(
  x = 1:N,
  intent = intent,
  g = g
) %>%
  ggplot(aes(x, intent)) +
  geom_point() +
  facet_wrap(~ g)

purchase_intent <- tibble(
  intent = intent,
  g = g
) %>% 
  bind_cols(X)

save(purchase_intent, file = here::here("Projects", "Data", "purchase_intent.RData"))

