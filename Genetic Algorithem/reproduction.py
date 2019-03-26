import random

import numpy as np
from cost_functions import CostFunction


def non_repeat_randint(low, high, size):

    # todo: check for inputs

    # a vector:
    if type(size) is int:
        list = []
        while len(list) < size:
            random_number = np.random.randint(low, high)
            if random_number not in list: list.append(random_number)
        return list

    # a matrix:
    matrix = np.zeros(size)
    for i in range(size[0]):
        list = np.random.uniform(0, 1, size[1])
        row = np.argsort(list)

        matrix[i, :] = row

    return matrix


class CrossingOver:

    def __init__(self, cost_fucntion: CostFunction, population_size: int = 50, cross_over_probability: float = 0.8,
                 mutation_probability: float = 0.2):
        """
        :param population_size: int - number of agents.
        :param cost_fucntion: CostFucntion - fitness_vector function.
        :param cross_over_probability: float - the probability of crossover, chance of an agents for crossing over.
        :param mutation_probability:
        """

        # get the aspects of problem:
        self.__dimension = cost_fucntion.get_dimensions()
        self.__max_boundary = cost_fucntion.get_max_boundary()
        self.__min_boundary = cost_fucntion.get_min_boundary()

        self.__sigma = 0.1 * (cost_fucntion.get_max_boundary() - cost_fucntion.get_min_boundary())  # use for mutation

        self.__population_size = population_size

        # the number of agents to participate in crossing over.
        # note we need two agents for a crossover, than number of agents for crossing over must be an even number.
        self.__agents_size_cv = 2 * round((population_size * cross_over_probability) / 2)

        self.__agents_size_mu = round(
            mutation_probability * population_size)  # number of agents under the influence of mutation.

    def get_cv_size(self):
        """
        The number of agents to participate in crossing over.
        :return:
        """
        return self.__agents_size_cv

    def get_mu_size(self):
        """
        The number of agents to participate in mutation.
        :return:
        """
        return self.__agents_size_mu

    def __single_point_crossover(self, parent1, parent2):
        cut_point = np.random.randint(low=1, high=self.__dimension - 1)

        child1 = np.append(parent1[0:cut_point], parent2[cut_point:])
        child2 = np.append(parent2[0:cut_point], parent1[cut_point:])
        return child1, child2

    def __double_point_crossover(self, parent1, parent2):
        cut_point1 = cut_point2 = 0
        while cut_point2 <= cut_point1:
            cut_point1 = np.random.randint(low=1, high=self.__dimension - 1)
            cut_point2 = np.random.randint(low=1, high=self.__dimension - 1)

        child1 = np.append(parent1[0:cut_point1], parent2[cut_point1:cut_point2])
        child1 = np.append(child1, parent1[cut_point2:])

        child2 = np.append(parent2[0:cut_point1], parent1[cut_point1:cut_point2])
        child2 = np.append(child2, parent2[cut_point2:])
        return child1, child2

    def __uniform_crossover(self, parent1, parent2):
        alpha = np.random.randint(0, 2)  # 1 or 0, randomly.

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def crossover(self, parent1, parent2):
        random_num = np.random.randint(1, high=4)

        if random_num == 1:
            return self.__single_point_crossover(parent1, parent2)  # dose't need bounding

        if random_num == 2:
            return self.__double_point_crossover(parent1, parent2)

        child1, child2 = self.__uniform_crossover(parent1, parent2)

        # boundary check:
        child1 = self.__check_fix_boundary(child1)
        child2 = self.__check_fix_boundary(child2)

        return child1.astype(int), child2.astype(int)

    def mutation(self, agent_row, mutation_rate: float = 0.6):

        if mutation_rate < 0 or mutation_rate > 1:
            raise Exception("Mutation rate must be between 0 and 1")

        # how many gene(input) of a chromosome should change?
        change_rate: int = round(mutation_rate * self.__dimension)

        indexes = non_repeat_randint(0, self.__dimension - 1, change_rate)  # get the chosen indices for change.
        agent = agent_row.copy()  # agent_row (call by reference).
        for index in indexes:
            agent[index] = agent[index] + self.__sigma * random.normalvariate(mu=0, sigma=1)  # just a formula.

        agent = np.array(agent).astype(int)
        return self.__check_fix_boundary(agent)  # return mutated agents, the new mutated population of agents.

    def get_population_size(self):
        return self.__population_size

    def __check_fix_boundary(self, agent_row):
        """
        check if the elements of a vector(agent row) it's between min boundary and max boundary.
        if it's not, fix it by assigning min or max boundary value to the element value.
        :param agent_row: numpy array - row vector
        :return: bounded vector.
        """
        return np.maximum(self.__min_boundary, np.minimum(self.__max_boundary, agent_row))
