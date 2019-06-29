# 29 May 2019 19:58
from cost_functions import *
import numpy as np

max_iteration = 100

# __init__:
cost_func = TravellingSalesmanProblem(10, 100)
swarm_size = 50
f = 0.5  # intensity of attraction
l = 1.5  # attractive length scale
# make the initial the population of grasshoppers in the response space:
swarm = np.random.uniform(cost_func.lower_bound, cost_func.upper_bound, size=[swarm_size, cost_func.dimensions])
swarm_costs = cost_func.compute_cost(swarm)

# save the best results:
position_history = []
cost_history = []
i = np.argmin(swarm_costs)
position_history.append(swarm[i].copy())
cost_history.append(swarm_costs[i])

target_position = swarm[i]
target_cost = swarm_costs[i]

c_max = 1
c_min = 0.00004


def distance(a, b):
    #              ___________________________________________________
    # d(a, b) =  √(a(1)-b(1))^2 + (a(2)-b(2))^2 + ... + (a(n)-b(n))^2
    d = (a - b) ** 2  # (subtracting element by element) ^ 2
    return np.sqrt(sum(d))


def s_func(r):
    return f * np.exp(-r / l) - np.exp(-r)  # Eq.(2.3) in the paper


for epoch in range(max_iteration):
    # def update_c (def, it:int):
    # update c using eq. 2.8:
    c = c_max - epoch * ((c_max - c_min) / max_iteration)

    # def change_grasshopper_position():
    for i in range(swarm_size):
        s_ij = 0
        for j in range(swarm_size):
            if i != j:
                distance_ij = distance(swarm[i], swarm[j])  # Calculate the distance between two grasshoppers
                r_ij_vec = (swarm[j] - swarm[i]) / (distance_ij + 1e-20)  # xj - xi / dij in Eq.(2.7)
                # xj_xi = np.abs(swarm[j] - swarm[i])  # | xjd - xid | in Eq.(2.7)
                xj_xi = cost_func.dimensions + (distance_ij % cost_func.dimensions)  # | xjd - xid | in Eq.(2.7) # ?

                # The first part inside the big bracket in Eq. (2.7):
                s_ij += ((cost_func.upper_bound - cost_func.lower_bound) * c / 2) * s_func(xj_xi) * r_ij_vec
        # update the grasshopper ith position:
        swarm[i] = c * s_ij + target_position  # Eq. (2.7) in the paper

    # boundary check:
    for i in range(len(swarm)):
        for j in range(cost_func.dimensions):  # <for each element(gene) in v, do>:
            # if element is more bigger that max boundary, set it to max boundary:
            swarm[i, j] = min(swarm[i, j], cost_func.upper_bound)
            # if element is less that min boundary, set it to min boundary:
            swarm[i, j] = max(swarm[i, j], cost_func.lower_bound)


    # update swarm costs:
    swarm_costs = cost_func.compute_cost(swarm)
    # save the best results:
    i = np.argmin(swarm_costs)
    position_history.append(swarm[i].copy())
    cost_history.append(swarm_costs[i])

    if target_cost > swarm_costs[i]:  # if <swarm_cost was better>:
        target_position = swarm[i]
        target_cost = swarm_costs[i]


cost_func.plot_cost_vs_iteration(cost_history)
i = np.argmin(cost_history)
cost_func.print_step_result(position_history[i], i)
cost_func.visual_result(position_history[i])




