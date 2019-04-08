import numpy as np
import matplotlib.pyplot as plt
from Selection import *
from Selection import RandomSelection
from reproduction import CrossingOver
from reproduction import non_repeat_randint
# problem definition:
# from cost_functions import Sphere
from cost_functions import TravellingSalesmanProblem

ff = TravellingSalesmanProblem(num_of_cities= 5, distance_range=100)  # fitness function
# ff.create_cities([10, 50, 30, 40, 10, 93, 20, 25], [5, 17, 80, 10, 12, 42, 14, 77])

# set Genetic parameters:
cv = CrossingOver(ff, population_size=100, mutation_rate=0.8)

# make the primitive population, create a (__population_size×__dimension) matrix with random numbers:
population = non_repeat_randint(low=ff.get_min_boundary(), high=ff.get_max_boundary(),
                                size=[cv.get_population_size(), ff.get_dimensions()])  # todo: changed the randint?
# compute fitness_vector function for primitive population (each agent in population):
fitness_vector = ff.compute_fitness(population).reshape([-1, 1])

# create a container for plotting the result:
max_iteration = 100
best_score = []  # save best fitness in each iteration

# randomSelection = RandomSelection(cv.get_population_size() - 1)  # indices start with zero
rouletteWheelSelection = RouletteWheelSelection(cv.get_population_size() - 1)
# main loop:
for iteration in range(max_iteration):

    'Crossover'
    # create a array to store new child agent in child population:
    child_population = np.zeros([cv.get_cv_size(), ff.get_dimensions()])

    for i in range(1, cv.get_cv_size(), 2):
        rouletteWheelSelection.make_roulette_wheel(fitness_vector)
        # find two parent to crossover with:
        idx_parent1 = rouletteWheelSelection.get_selected()
        idx_parent2 = rouletteWheelSelection.get_selected()
        # crossover and save the new population, in the array list:
        child_population[i - 1], child_population[i] = cv.crossover(population[idx_parent1],
                                                                    population[idx_parent2])

    # compute the fitness_vector for the new population (children) in a column array:
    child_fitness_vector = ff.compute_fitness(child_population).reshape([-1, 1])

    'Merge'
    population = np.append(population, child_population, axis=0)
    fitness_vector = np.append(fitness_vector, child_fitness_vector)
    # sort base on the fitness_vector function (dec)
    # throw away useless agents by getting only the indexes of top of the sorted array.
    ans = np.argsort(fitness_vector, axis=0)[:cv.get_population_size()]

    i = 0
    sorted_population = np.zeros([cv.get_population_size(), ff.get_dimensions()])
    sorted_fitness = np.zeros(cv.get_population_size())
    for index in ans:
        sorted_population[i, :] = population[index, :]
        sorted_fitness[i] = fitness_vector[index]
        i += 1
    population = sorted_population
    fitness_vector = sorted_fitness
    best_score.append(fitness_vector[0])

    if iteration % 5 == 0:
        print("iteration", iteration, "= ", population[0], " FF:{}".format(fitness_vector[0]))
        ff.plot_agent_travel_order(population[0])

plt.plot(range(0, len(best_score)), best_score)
plt.xlabel("iteration")
plt.ylabel("Fitness")
plt.show()
