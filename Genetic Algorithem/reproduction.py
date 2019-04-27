import random

import numpy as np
from cost_functions import CostFunction


def non_repeat_randint(low, high, size):
    """
    by Hussein Asshabi
    last update: 17 April 2019
    """
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
    """
    by Hussein Asshabi
    last update: 18 April 2019
    """

    def __init__(self, cost_function: CostFunction, population_size: int = 100, mutation_rate=0.1,
                 cross_over_probability: float = 1.0):
        """
        :param population_size: int - number of agents.
        :param cost_function: CostFunction - fitness_vector function.
        :param mutation_rate:
        :param cross_over_probability: float - the probability of crossover, chance of an agents for crossing over.
        """

        # get the aspects of problem:
        self.__cost_function = cost_function
        self.__population_size = population_size
        self.__mutation_rate = mutation_rate

        # the number of agents to participate in crossing over.
        # note we need two agents for a crossover, than number of agents for crossing over must be an even number.
        self.__agents_size_cv = 2 * round((population_size * cross_over_probability) / 2)

    def get_cv_size(self):
        """
        :return: the number of agents to participate in crossing over.
        """
        return self.__agents_size_cv

    def __single_point_crossover(self, parent1, parent2):
        cut_point = np.random.randint(low=1, high=self.__cost_function.get_dimensions() - 1)

        child1 = np.append(parent1[0:cut_point], parent2[cut_point:])
        child2 = np.append(parent2[0:cut_point], parent1[cut_point:])
        return child1, child2

    def __double_point_crossover(self, parent1, parent2):
        cut_point1 = cut_point2 = 0
        while cut_point2 <= cut_point1:
            cut_point1 = np.random.randint(low=1, high=self.__cost_function.get_dimensions() - 1)
            cut_point2 = np.random.randint(low=1, high=self.__cost_function.get_dimensions() - 1)

        child1 = np.append(parent1[0:cut_point1], parent2[cut_point1:cut_point2])
        child1 = np.append(child1, parent1[cut_point2:])

        child2 = np.append(parent2[0:cut_point1], parent1[cut_point1:cut_point2])
        child2 = np.append(child2, parent2[cut_point2:])
        return child1, child2

    def __uniform_crossover(self, parent1, parent2):
        alpha = np.random.randint(0, 2)  # 1 or 0, randomly.

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1

        # boundary check:
        child1 = self.__check_fix_boundary(child1)
        child2 = self.__check_fix_boundary(child2)
        return child1, child2

    def crossover(self, parent1, parent2):
        cut_point_chooser = np.random.randint(1, high=4)

        if cut_point_chooser == 1:
            child1, child2 = self.__single_point_crossover(parent1, parent2)  # dose't need bounding

        if cut_point_chooser == 2:
            child1, child2 = self.__double_point_crossover(parent1, parent2)

        child1, child2 = self.__uniform_crossover(parent1, parent2)

        # mutation on children:
        child1 = self.mutation(agent_row=child1)
        child2 = self.mutation(agent_row=child2)

        return child1, child2

    def mutation(self, agent_row):
        """

        :param agent_row:
        :return:
        """
        '''
        for i in range(0, len(agent_row)):
            if np.random.rand() < self.__mutation_rate:
                # change the gene to a new value:
                agent_row[i] = np.random.randint(self.__cost_function.get_min_boundary(),
                                                 self.__cost_function.get_max_boundary())
        '''
        for i in range(0, len(agent_row)):
            if np.random.rand() < self.__mutation_rate:
                # change the gene to a new value:
                idx1 = np.random.randint(len(agent_row))
                idx2 = np.random.randint(len(agent_row))

                # swap index1 with index2:
                temp = agent_row[idx1]
                agent_row[idx1] = agent_row[idx2]
                agent_row[idx2] = temp

        return agent_row

    def get_population_size(self):
        return self.__population_size

    def __check_fix_boundary(self, agent_row):
        """

        Check if the elements of a vector(agent row) it's between min boundary and max boundary.
        if it's not, fix it by assigning min or max boundary value to the element value.
        :param agent_row: numpy array - row vector
        :return: bounded vector.
        """
        return np.maximum(self.__cost_function.get_min_boundary(),
                          np.minimum(self.__cost_function.get_max_boundary(), agent_row))
