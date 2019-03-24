import numpy as np
from Selection import RandomSelection


def single_point_crossover(parent1, parent2):
    n = len(parent1)
    cut_point = np.random.randint(low=1, high=n - 1)
    child1 = np.append(parent1[0:cut_point], parent2[cut_point:])
    child2 = np.append(parent2[0:cut_point], parent1[cut_point:])
    return child1, child2


def double_point_crossover(parent1, parent2):
    n = len(parent1)

    cut_point1 = cut_point2 = 0
    while cut_point2 <= cut_point1:
        cut_point1 = np.random.randint(low=1, high=n - 1)
        cut_point2 = np.random.randint(low=1, high=n - 1)

    child1 = np.append(parent1[0:cut_point1], parent2[cut_point1:cut_point2])
    child1 = np.append(child1, parent1[cut_point2:])

    child2 = np.append(parent2[0:cut_point1], parent1[cut_point1:cut_point2])
    child2 = np.append(child2, parent2[cut_point2:])
    return child1, child2


def crossover(parent1, parent2):
    s = np.random.randint(1, high=3)

    if s == 1:
        return single_point_crossover(parent1, parent2)

    if s == 2:
        child1, child2 = double_point_crossover(parent1, parent2)
        return child1, child2

    child1, child2 = uniform_crossover(parent1, parent2)
    return child1, child2


def uniform_crossover(parent1, parent2):
    n = len(parent1)
    alpha = np.random.randint(0, 2)

    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


# problem definition:
#  sphere Cost Function:
from CostsFunction import sphere as cost_function

min_boundary = -10
max_boundary = 10
dimension = 5

# Genetic parameters:
num_agents = 1000  # [set by user]
prob_crossover = 0.8  # the probability of crossover, chance of an agents to crossover. [set by user]
# the number of agents to participate in crossovering.
# note we need two agents for a crossover, than number of agents for crossovering must be an even number.
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

# main loop:
for it in range(max_iteration):

    randomSelection = RandomSelection(num_agents_cv)

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


    def non_repeat_randint(low, high, size):
        # todo: check for inputs
        import random
        list = []
        while len(list) < size:
            random_number = random.randint(low, high)
            if random_number not in list: list.append(random_number)

        return list


    def mutation(agent, mutation_rate):
        import random

        if mutation_rate < 0 or mutation_rate > 1:
            raise Exception("Mutation rate must be between 0 and 1")

        dimension = len(agent)
        step_size = 0.1 * (max_boundary - min_boundary)

        num_mutation = round(mutation_rate * dimension)  # how many gene(input) of a chromosome should change?
        indexes = non_repeat_randint(0, dimension - 1, num_mutation)

        for index in indexes:
            agent[index] = agent[index] + step_size * random.normalvariate(mu=0, sigma=1)  # just a formula.

        mutated = agent
        return mutated


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
    # throw away uselesses
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