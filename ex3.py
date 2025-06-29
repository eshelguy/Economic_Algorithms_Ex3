#Guy Eshel
#208846758

import numpy as np
import cvxpy as cp

def compute_feasible_optimal_bundles(value_matrix, prices, budgets, supply=None):
    n, m = value_matrix.shape

    # Default: 1 unit of each resource
    if supply is None:
        supply = np.ones(m)

    x = cp.Variable((n, m), nonneg=True)

    # Objective: maximize total utility
    objective = cp.Maximize(cp.sum(cp.multiply(value_matrix, x)))

    constraints = []

    # Budget constraint for each player
    for i in range(n):
        constraints.append(prices @ x[i, :] <= budgets[i])

    # Supply constraint for each resource
    for j in range(m):
        constraints.append(cp.sum(x[:, j]) <= supply[j])

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value

def print_allocations(allocation, value_matrix, prices, budgets):
    n, m = allocation.shape
    for i in range(n):
        print(f"Player {i+1} receives:")
        for j in range(m):
            print(f"  Resource {j+1}: {allocation[i][j]:.4f}")
        cost = np.dot(prices, allocation[i])
        value = np.dot(value_matrix[i], allocation[i])
        print(f"  Total value: {value:.2f}")
        print(f"  Total cost:  {cost:.2f} (Budget: {budgets[i]})\n")


if __name__ == "__main__":
    print("Test Case 1: class example")
    value_matrix = np.array([
        [8, 4, 2],  # Player 1
        [2, 6, 5]   # Player 2
    ])
    prices = np.array([52.2, 26.1, 21.7])
    budgets = np.array([60, 40])

    allocation = compute_feasible_optimal_bundles(value_matrix, prices, budgets)
    print_allocations(allocation, value_matrix, prices, budgets)

    print("Test Case 2: Single man")
    value_matrix2 = np.array([
        [10, 3, 0]
    ])
    prices2 = np.array([5, 5, 5])
    budgets2 = np.array([10])

    allocation2 = compute_feasible_optimal_bundles(value_matrix2, prices2, budgets2)
    print_allocations(allocation2, value_matrix2, prices2, budgets2)

    print("Test Case 3: Balanced Market")
    value_matrix3 = np.array([
        [3, 5, 7],
        [7, 5, 3]
    ])
    prices3 = np.array([10, 10, 10])
    budgets3 = np.array([20, 20])

    allocation3 = compute_feasible_optimal_bundles(value_matrix3, prices3, budgets3)
    print_allocations(allocation3, value_matrix3, prices3, budgets3)

    print("Test Case 4: Expensive Favorite")
    value_matrix4 = np.array([
        [9, 1, 1]
    ])
    prices4 = np.array([100, 10, 10])
    budgets4 = np.array([50])

    allocation4 = compute_feasible_optimal_bundles(value_matrix4, prices4, budgets4)
    print_allocations(allocation4, value_matrix4, prices4, budgets4)

    print("Test Case 5: Three Players")
    value_matrix5 = np.array([
        [5, 5, 5],
        [9, 1, 1],
        [2, 3, 8]
    ])
    prices5 = np.array([10, 10, 10])
    budgets5 = np.array([15, 20, 30])

    allocation5 = compute_feasible_optimal_bundles(value_matrix5, prices5, budgets5)
    print_allocations(allocation5, value_matrix5, prices5, budgets5)

    print("Test Case 6: Equal Preferences")
    value_matrix6 = np.array([
        [3, 4, 3],
        [3, 4, 3]
    ])
    prices6 = np.array([20, 10, 40])
    budgets6 = np.array([20, 80])

    allocation6 = compute_feasible_optimal_bundles(value_matrix6, prices6, budgets6)
    print_allocations(allocation6, value_matrix6, prices6, budgets6)