from cost_functions import *
import numpy as np


# initialize parameters:
cost_func = Sphere()  # problem definition.
dim = cost_func.dimensions
lb = cost_func.lower_bound
ub = cost_func.upper_bound

max_iteration = 500
n_whales = 50
b = 3  # a constant, parameters in Eq.(2.5)

# Initialize the positions of search agents:
positions = np.random.uniform(lb, ub, size=[n_whales, dim])
scores = cost_func.compute_cost(positions)  # Compute score(fitness) for all whales

# Initialize position and score for the leader (or the bait):
index = np.argmin(scores)  # find the lowest (best) score.

# Set the best whale as the leader:
leader_position = positions[index]
leader_score = scores[index]

# for sav1ing the result of each iteration:
position_history = []
cost_history = []

for t in range(max_iteration):
    # Update a (for all whale), A, C, l, and p (for each whale):
    a = 2 - t * (2 / max_iteration)  # a decreases linearly from 2 to 0 in Eq. (2.3)

    for i in range(n_whales):
        r1 = np.random.rand() # r1 is a random number in [0, 1]
        r2 = np.random.rand() # r2 is a random number in [0, 1]

        A = 2 * a * r1 - a  # Eq.(2.3)
        C = 2 * r2  # Eq.(2.4)

        l = np.random.uniform(-1, 1) # parameters in Eq.(2.5)
        p = np.random.rand() # p in Eq.(2.6)

        if p < 0.5:
            if abs(A) >= 1: # Search for prey randomly (exploration phase)
                # Select a random search agent (x_rand or rand_whale) a random index, choosing a whale randomly):
                rand_index = np.random.randint(n_whales)  # a random number [0, n_whales]
                rand_whale = positions[rand_index].copy()
                # Update the position of the current search agent by the Eq. (2.8):
                D = abs(C * rand_whale - positions[i]) # Eq. (2.7)
                positions[i] = rand_whale - A * D  # Eq. (2.8) changing the position of the i_th wheal.
            else:  # |A| < 1:
                # Update the position of the current search agent by the Eq. (2.1):
                D =  abs(C * leader_position - positions[i]) # Eq. (2.1)
                positions[i] = leader_position - A * D  # Eq. (2.2)

        else: # p < 0.5: Spiral updating position. Fig. 4 (b):
            # Update the position of the current search by the Eq. (2.5):
            D = abs(leader_position - positions[i])  # distance to the leader whale.
            positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_position # Eq. (2.5)

    # Check if any search agent goes beyond the search space and amend it:
    for i in range(n_whales):
        for j in range(cost_func.dimensions):  # <for each element(gene) in v, do>:
            # if element is more bigger that max boundary, set it to max boundary:
            positions[i, j] = min(positions[i, j], cost_func.upper_bound)
            # if element is less that min boundary, set it to min boundary:
            positions[i, j] = max(positions[i, j], cost_func.lower_bound)

    # Calculate the fitness of each search agent:
    scores = cost_func.compute_cost(positions)
    # Update X* if there is a better solution:
    index = np.argmin(scores)  # find the lowest (best) score.

    # Set the best whale as the leader:
    if leader_score > scores[index]:  # if <the best current whale had a better position then the leader>:
        leader_position = positions[index].copy()
        leader_score = scores[index].copy()


    # save the best results of the current iteration:
    position_history.append(leader_position.copy())
    cost_history.append(leader_score.copy())

cost_func.plot_cost_vs_iteration(cost_history)
index = np.argmin(cost_history)
iteration = index

cost_func.print_step_result(position_history[index], iteration)
cost_func.visual_result(position_history[index])