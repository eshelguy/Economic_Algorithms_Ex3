#Guy Eshel
#208846758

import cvxpy
import numpy

def egalitarian_allocation(values):
    num_agents, num_resources = values.shape

    # Define the decision variables
    allocations = cvxpy.Variable((num_agents, num_resources))

    # Perceived value for each agent
    perceived_value = cvxpy.sum(cvxpy.multiply(allocations, values), axis=1)

    # The minimum value to be maximized
    min_utility = cvxpy.Variable()

    # Define the problem
    prob = cvxpy.Problem(cvxpy.Maximize(min_utility),
                         constraints=[
                             cvxpy.sum(allocations, axis=0) == 1,  # The sum of allocations for each resource must be 1
                             allocations >= 0,  # Allocations cannot be negative
                             perceived_value >= min_utility  # Ensuring fairness by maximizing the smallest perceived value
                         ])
    prob.solve()

    print(f"Maximum egalitarian value: {min_utility.value:.2f}")
    for i in range(num_agents):
        print(f"Agent {i + 1} receives:")
        for j in range(num_resources):
            print(f"  {allocations.value[i, j]:.2f} of resource #{j + 1}")
        print()


if __name__ == "__main__":
    print("Test Case 1: Basic Case - our example")
    values = numpy.array([[80, 19, 1],
                       [70, 1, 29]])
    egalitarian_allocation(values)

    print("\nTest Case 2: Equal Valuations")
    values = numpy.array([[50, 50, 50],
                       [50, 50, 50],
                       [50, 50, 50]])
    egalitarian_allocation(values)

    print("\nTest Case 3: Single Resource")
    values = numpy.array([[100],
                       [50],
                       [20]])
    egalitarian_allocation(values)

    print("\nTest Case 4: Highly Skewed Valuations")
    values = numpy.array([[1000, 1, 1, 1],
                       [1, 1, 1, 1000]])
    egalitarian_allocation(values)

    print("\nTest Case 5: Zero Valuation for Some Resources")
    values = numpy.array([[50, 0, 10],
                       [30, 40, 0],
                       [0, 100, 0]])
    egalitarian_allocation(values)

    print("\nTest Case 6: Large Number of Agents and Resources")
    values = numpy.array([[1, 2, 3, 4, 5],
                       [5, 4, 3, 2, 1],
                       [2, 2, 2, 2, 2],
                       [1, 1, 1, 1, 1],
                       [3, 3, 3, 3, 3]])
    egalitarian_allocation(values)

    print("\nTest Case 7: Equal Valuations")
    values = numpy.array([[90, 10],
                          [50, 50],
                          [10, 90]])
    egalitarian_allocation(values)