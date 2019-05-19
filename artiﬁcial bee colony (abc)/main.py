import pandas as pd
import numpy as np
from cost_functions import *
from selection_method import RouletteWheelSelection


def flying_to_food_source(i):
    # choose a random food source j ≠ k:
    k = i
    while k == i:
        k = np.random.randint(0, n_employed)
    # In order to produce a candidate food position from the old one, the ABC uses the
    # following expression (2.2):
    # vij = xij + φij(xij − xkj)  where k ∈ {1, 2, . . . , n_employed} and j ∈ {1, 2, . . . , dimension}
    phi = np.random.uniform(-1, 1, [1, cost.dimensions])
    v = (employees[i] + phi * (employees[k] - employees[i])).flatten()
    # boundary check:
    for j in range(len(v)):  # <for each element(gene) in v, do>:
        v[j] = min(v[j], cost.max_boundary)  # if element is more bigger that max boundary, set it to max boundary.
        v[j] = max(v[j], cost.min_boundary)  # if element is less that min boundary, set it to min boundary.
    # compute ith bee cost:
    v_cost = cost.compute_cost(v)
    if v_cost < employees_cost[i]:  # <if v has better cost do:>
        # replace it with ith employee:
        employees[i] = v
        employees_cost[i] = v_cost
    else:
        limits[i] = limits[i] - 1  # give a negative feedback.


# initialize:
cost = NQueen(20)  # The problem to solve!
max_iteration = 200

swarm_size = 20
n_onlookers = round(swarm_size / 2)  # 50% or half of the swarm.
n_employed = n_onlookers  # same as number of onlookers.

limit = n_onlookers * cost.dimensions  # If a solution representing a food source
#   is not improved by a predetermined number of trials, then that food source is abandoned by its employed bee.

# make the initialize population (send scout bees to find the food sources):
employees = np.random.uniform(low=cost.min_boundary, high=cost.max_boundary, size=[swarm_size, cost.dimensions])
employees_cost = cost.compute_cost(employees)  # compute cost (fitness) of the employee bees.

# to keep track and saving the best solution, we define the following:
index = np.argmin(employees_cost)  # take the index of the best cost.
top_food_source = pd.DataFrame({'index': index,
                                'position': [employees[index]],
                                'cost': employees_cost[index]})
# top_food_source.get_values()[-1][-1]
'main loop:'
limits = np.array([limit] * swarm_size)  # for each food source we have a limit.

# <use following code for while loop instead of for loop>:
iteration = 0
threshold = 4
best_cost = np.inf
while best_cost > threshold:
    iteration += 1
# <end>

# for iteration in range(max_iteration):
    #  (a) Place the employed bees on the food sources in the memory:
    # send employed bee to food sources that scout bee found (note that we didn't define
    #   scout bees and we treat employed bee as both scout and employed bee):
    for index in range(n_employed):
        flying_to_food_source(index)

    # (b) Place the onlooker bees on the food sources in the memory:
    # An onlooker bee chooses a food source depending on the probability value associated with that food source,
    #   pi, calculated by the following expression (2.1):
    # pi = ﬁti / Σ ﬁtn  where ﬁti is the ﬁtness value of the solution i evaluated by its employed bee.
    # employees_cost_reverse = max(
    #     employees_cost) - employees_cost  # due minimization, we need more probability for a small cost.
    # probabilities = employees_cost_reverse / sum(employees_cost_reverse)

    # roulette_wheel:
    rws = RouletteWheelSelection()
    rws.make_roulette_wheel(employees_cost)

    for index in range(n_onlookers):  # <for each food source do:>
        flying_to_food_source(rws.roll())

    # (c) Send the scouts to the search area for discovering new food sources:
    for i in range(n_employed):
        if limits[i] <= 0:  # <if it's not nectar in a such food source i, do:>
            limits[i] = limit  # reset the counter.
            employees[i] = np.random.uniform(cost.min_boundary, cost.max_boundary,
                                             cost.dimensions)  # find another food source.
            employees_cost[i] = cost.compute_cost(employees[i])

    index = np.argmin(employees_cost)  # take the index of the best cost.
    top_food_source = top_food_source.append({'index': index,
                                              'position': [employees[index]],
                                              'cost': employees_cost[index]}, ignore_index=True)

    # <use following code for while loop instead of for loop>:
    best_cost = min(employees_cost)
    # <end>

cost.plot_cost_iteration(top_food_source.cost)
solution = cost.get_print_solution(top_food_source.position, top_food_source.cost)
cost.plot_solution(solution)
# this algorithm it doesnt fall into local minimal.
# it dose exploration and exploitation at the same time.
