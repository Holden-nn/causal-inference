# My Project Report


## Data Story

Lorem ipsum odor amet, consectetuer adipiscing elit. Sit eu class
placerat erat nostra. Dictum sed nibh amet vel sollicitudin curabitur
himenaeos ante. Gravida risus conubia per ultrices ligula nascetur. Nam
fusce amet ad ante sed aenean! Quis lacus vulputate bibendum facilisi
condimentum sit.

Mauris volutpat iaculis enim nam taciti est ipsum dui. Eu facilisi erat
euismod per nulla lacus. Neque duis bibendum conubia class fusce fames.
Ultrices ultricies in hac ornare platea. Aliquam sociosqu habitant
conubia porta sagittis sociosqu aenean? Nunc eget accumsan nunc lacus,
urna lacus? Est quisque quis iaculis phasellus nisl. Purus nisi cursus
convallis, tristique mauris sagittis nibh.

## DAG

``` mermaid
flowchart LR
  A(A) --> B(B)
  A --> C(C)
  C --> B
  D(D) --> Y
  B --> Y{Y}
```

## Identification Strategy

Here’s my adjustment set:

- Lorem ipsum odor amet, consectetuer adipiscing elit.
- Euismod a inceptos torquent laoreet dapibus quis quam laoreet.
- Magnis lacinia ante aliquam posuere parturient lobortis.

## Simulate Data and Recover Parameters

``` python
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Set the parameter values.
beta0 = 3
beta1 = 7
n = 100

sim_data = (
    # Simulate predictors using appropriate np.random distributions.
    pl.DataFrame({
        'x': np.random.uniform(0, 7, size = n)
    })
    # Use predictors and parameter values to simulate the outome.
    .with_columns([
        (beta0 + beta1 * pl.col('x') + np.random.normal(0, 3, size = n)).alias('y')
    ])
)

sim_data

# Specify the X matrix and y vector.
X = sim_data[['x']]
y = sim_data['y']

# Create a linear regression model.
model = LinearRegression(fit_intercept=True)

# Train the model.
model.fit(X, y)

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')

# Have you recovered the parameters?
```

    Intercept: 3.6452884726402353
    Slope: 6.802954331232987

Lorem ipsum odor amet, consectetuer adipiscing elit. Sagittis interdum
fringilla sagittis platea eget dictum sodales non. Nec arcu porta felis
eros sem accumsan? Sit quis ridiculus, ligula dictum ex luctus.

## Exploratory Data Analysis
