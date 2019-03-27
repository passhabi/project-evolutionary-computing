import numpy as np
import matplotlib.pyplot as plt
from Selection import RandomSelection
from reproduction import CrossingOver
from reproduction import non_repeat_randint
# problem definition:
# from cost_functions import Sphere
from cost_functions import TravellingSalesmanProblem

ff = TravellingSalesmanProblem(10, 100)  # fitness function

# set Genetic parameters:
cv = CrossingOver(ff, population_size=100, mutation_probability=0.9)

# make the primitive population, create a (__population_size×__dimension) matrix with random numbers:
population = non_repeat_randint(low=ff.get_min_boundary(), high=ff.get_max_boundary(),
                                size=[cv.get_population_size(), ff.get_dimensions()])  # todo: changed the randint?
# compute fitness_vector function for primitive population (each agent in population):
fitness_vector = ff.compute_fitness(population).reshape([-1, 1])

# create a container for plotting the result:
max_iteration = 551
best_fitness = np.zeros(max_iteration)  # save best fitness in each iteration

randomSelection = RandomSelection(cv.get_population_size() - 1)  # indices start with zero
# main loop:
for iteration in range(max_iteration):

    'Crossover'
    # create a array to store new child agent in child population:
    child_population = np.zeros([cv.get_cv_size(), ff.get_dimensions()])

    for i in range(1, cv.get_cv_size(), 2):
        # find two parent to crossover with:
        index_parent1 = randomSelection.get_selected()
        index_parent2 = randomSelection.get_selected()
        # crossover and save the new population, new agents (children) in the child_population array list:
        child_population[i - 1], child_population[i] = cv.crossover(population[index_parent1],
                                                                    population[index_parent2])

    # compute the fitness_vector for the new population (children) in a column array:
    child_fitness_vector = ff.compute_fitness(child_population).reshape([-1, 1])

    'Mutation'
    mutated_population = np.zeros([cv.get_mu_size(), ff.get_dimensions()])  # create a array to store new mutated agent.

    for i in range(0, cv.get_mu_size()):
        # find an agent for mutation:
        agent_index = randomSelection.get_selected()
        # mutation and save the new population, new agents (mutated) in the mutated_population array list:
        mutated_population[i] = cv.mutation(population[agent_index])

    # compute the fitness_vector for the new population (mutated) in a column array:
    mutated_fitness_vector = ff.compute_fitness(mutated_population).reshape([-1, 1])

    'Merge'
    population = np.append(np.append(population, child_population, axis=0), mutated_population, axis=0)
    fitness_vector = np.append(np.append(fitness_vector, child_fitness_vector), mutated_fitness_vector).reshape(-1, 1)
    # sort base on the fitness_vector function (dec)
    # throw away useless agents by getting only the indexes of top of the sorted array.
    ans = np.argsort(fitness_vector, axis=0)[:cv.get_population_size()]

    #
    i = 0
    sorted_population = np.zeros([cv.get_population_size(), ff.get_dimensions()])
    sorted_fitness = np.zeros(cv.get_population_size())
    for index in ans:
        sorted_population[i, :] = population[index, :]
        sorted_fitness[i] = fitness_vector[index]
        i += 1
    population = sorted_population
    fitness_vector = sorted_fitness

    if iteration % 5 == 0:
        print("iteration", iteration, "(", population[0], " FF:{}".format(fitness_vector[0]), ")")
    best_fitness[iteration] = fitness_vector[0]

plt.plot(range(0, max_iteration), best_fitness)
plt.xlabel("iteration")
plt.ylabel("Fitness")
plt.show()

ff.plot_agent_travel_order(population[0])
