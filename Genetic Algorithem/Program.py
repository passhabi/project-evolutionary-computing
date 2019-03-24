import numpy as np
from Selection import RandomSelection

# problem definition:
#  sphere Cost Function:
from cost_functions import sphere as cost_function

min_boundary = -10
max_boundary = 10
dimension = 5

# Genetic parameters:
num_agents = 1000  # [set by user]
prob_crossover = 0.8  # the probability of crossover, chance of an agents to crossover. [set by user]
# the number of agents to participate in crossing over.
# note we need two agents for a crossover, than number of agents for crossing over must be an even number.
num_agents_cv = 2 * round((num_agents * prob_crossover) / 2)

prob_mutation = 0.2
num_agents_mu = round(prob_mutation * num_agents)  # number of agents under the influence of mutation
mutation_rate = 0.6  # percent of mutation.

# make the primitive population:cdgit
population = np.random.uniform(min_boundary, max_boundary,
                               [num_agents, dimension])  # make a (num_agents*dimension) matrix with random numbers.
fitness = np.sum(population ** 2, axis=1).reshape([num_agents, 1])  # compute fitness of each agent in population.

# create a container for keeping the best agent for plotting.
# best_index = np.argmin(fitness)
# best_fitness = np.min(fitness)
max_iteration = 100
error = np.zeros(max_iteration)

randomSelection = RandomSelection(num_agents_cv)
from reproduction import crossover, mutation

# main loop:
for it in range(max_iteration):

    'Crossover'
    child_population = np.zeros([num_agents_cv, dimension])  # create a array to store new child agent.
    for i in range(1, num_agents_cv, 2):
        # find two parent to crossover with:
        index_parent1 = randomSelection.get_selected()
        index_parent2 = randomSelection.get_selected()
        # crossover and save the new population, new agents (children) in the child_population array list:
        child_population[i - 1], child_population[i] = crossover(population[index_parent1], population[index_parent2])

    # compute the fitness for the new population (children) in a column array:
    child_fitness = np.sum(child_population ** 2, axis=1).reshape([num_agents_cv, 1])
    # todo: boundary check.
    'Mutation'
    mutated_population = np.zeros([num_agents_mu, dimension])  # create a array to store new mutated agent.



    del randomSelection
    randomSelection = RandomSelection(num_agents_mu)
    for i in range(0, num_agents_mu):
        # find an agent for mutation:
        agent_index = randomSelection.get_selected()
        # mutation and save the new population, new agents (mutated) in the mutated_population array list:
        mutated_population[i] = mutation(population[agent_index], mutation_rate)

    # compute the fitness for the new population (mutated) in a column array:
    mutated_fitness = np.sum(mutated_population ** 2, axis=1).reshape([num_agents_mu, 1])

    'Merge'
    population = np.append(np.append(population, child_population, axis=0), mutated_population, axis=0)
    fitness = np.append(np.append(fitness, child_fitness), mutated_fitness).reshape(-1, 1)
    # sort base on the fitness function (dec)
    # throw away useless agents
    ans = np.argsort(fitness, axis=0)[:num_agents]  # get index of first (num_agents: int) sorted agent.
    iii = 0
    sorted_population = np.zeros([num_agents, dimension])
    sorted_fitness = np.zeros(num_agents)
    for index in ans:
        sorted_population[iii] = population[index]
        sorted_fitness[iii] = fitness[index]
        iii += 1
    population = sorted_population
    fitness = sorted_fitness

    error[it] = fitness[0]

import matplotlib.pyplot as plt

plt.plot(range(0, max_iteration), error)
plt.xlabel("iteration")
plt.ylabel("Fitness")
plt.show()