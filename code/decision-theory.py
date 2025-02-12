import numpy as np

# Define a linear demand function.
def demand(price):
    demand = max(alpha - beta * price, 0)
    return demand

# Define a simple profit function.
def profit(price):
    profit = (price - cost) * demand(price)
    return profit

alpha = 100 # Assume a base demand.
beta = 5    # Assume price sensitivity.
cost = 10   # Assume fixed cost.

# Possible prices.
prices = np.array([5, 10, 15, 20, 25])

np.array([profit(price) for price in prices])  # Maximize profit.
-np.array([profit(price) for price in prices]) # Minimize loss.

