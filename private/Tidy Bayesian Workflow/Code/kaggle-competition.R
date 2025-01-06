# Preamble ----------------------------------------------------------------
# Load packages.
library(rethinking)
library(tidyverse)
library(tidymodels)
library(rstan)
library(bayesplot)
library(tidybayes)

# Set simulation seed.
set.seed(42)

# Stan options.
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Import and mutate data.
data <- read_csv(here::here("Projects", "Data", "data.csv")) %>%
  separate(game_date, into = c("year", "month", "day")) %>%            # Get year, month, and day out of game_date.
  mutate(
    time_remaining = str_c(minutes_remaining, ".", seconds_remaining), # Create a time_remaining variable.
    time_remaining = as.numeric(time_remaining),                       # Make sure it's numeric.
    home = str_detect(matchup, "vs."),                                 # Create a home dummy.
    home = as.numeric(home),                                           # Make sure it's numeric.
    period = as.character(period),                                     # Make sure period is nominal.
    playoffs = as.numeric(playoffs)                                    # Make sure playoffs is numeric.
  ) %>%
  select(
    -c(
      team_id, team_name,                                              # No variation (always Lakers).
      game_event_id, game_id,                                          # Different IDs.
      shot_id,                                                         # Shot ID that is duplicated in shot type variables.
      matchup,                                                         # Variable that we extracted the home game dummy from.
      lat, lon,                                                        # Latitude and longitude (we're using loc_x and loc_y).
      minutes_remaining, seconds_remaining                             # Variables used to create time_remaining.
    )
  )

# Split into training and testing.
train <- data %>% filter(!is.na(shot_made_flag))
test <- data %>% filter(is.na(shot_made_flag)) %>% select(-shot_made_flag)

# Split training into training and validation.
train_split <- initial_split(train, prop = 0.8)

# Prepare the data for modeling.
data_recipe <- training(train_split) %>%                       # Create a recipe for data prep.
  recipe(shot_made_flag ~ .) %>%                               # Specify model.
  step_center(loc_x, loc_y, time_remaining, shot_distance) %>% # Center variables.
  step_scale(loc_x, loc_y, time_remaining, shot_distance) %>%  # Scale variables.
  step_dummy(all_nominal()) %>%                                # Create dummies out of all_nominal() variables.
  prep()                                                       # Execute recipe on the training data.

# Extract the prepped training data.
train_partial <- juice(data_recipe)

# Apply the recipe to the validation data and drop rows with NAs.
valid_partial <- data_recipe %>% 
  bake(testing(train_split)) %>% 
  drop_na()

# Apply the recipe to the test data for prediction and replace NAs with 0s.
test_baked <- data_recipe %>% 
  bake(test) %>%
  map_df(replace_na, 0)

# Flat Model --------------------------------------------------------------
# Specify data.
data_list <- list(
  # Covariate index.
  N_train = nrow(train_partial),
  N_valid = nrow(valid_partial),
  N_test = nrow(test_baked),
  I = ncol(train_partial),
  
  # Vector of training data observations.
  shot_made_flag = train_partial$shot_made_flag,
  
  # Matrix of training data covariates.
  X_train = train_partial %>%
    mutate(intercept = 1) %>%
    select(-shot_made_flag),
  
  # Matrix of validation data covariates.
  X_valid = valid_partial %>%
    mutate(intercept = 1) %>%
    select(-shot_made_flag),
  
  # Matrix of testing data covariates.
  X_test = test_baked %>%
    mutate(intercept = 1)
)

# Fit the model and save output.
out03 <- stan(
  file = here::here("Projects", "Code", "model01.stan"),
  data = data_list,
  seed = 42
)
save(out03, file = here::here("Projects", "Output", "model03-output.RData"))

# Extract predictions.
y_valid <- rstan::extract(out03)$y_valid
p_test <- rstan::extract(out03)$p_test

# Check with validation.
valid_partial %>% 
  select(shot_made_flag) %>% 
  mutate(y_valid = ifelse(apply(y_valid, 2, mean) > 0.5, 1, 0)) %>% 
  mutate(hit = abs(shot_made_flag - y_valid)) %>% 
  summarize(
    loss = mean(hit),
    hit_rate = 1 - mean(hit)
  )

# Prepare submission.
read_csv(here::here("Projects", "Data", "sample_submission.csv")) %>% 
  select(shot_id) %>% 
  mutate(shot_made_flag = apply(p_test, 2, mean)) %>% 
  write_csv(here::here("Projects", "Output", "submission.csv"))

