import random
import numpy as np
from cost_functions import CostFunction


def non_repeat_randint(low, high, size):
    # todo: check for inputs
    list = []
    while len(list) < size:
        random_number = random.randint(low, high)
        if random_number not in list: list.append(random_number)
    return list


class CrossingOver:

    def __init__(self, cost_fucntion: CostFunction):

        # get the aspects of problem:
        self.dimension = cost_fucntion.get_dimensions()
        self.max_boundary = cost_fucntion.get_max_boundary()
        self.min_boundary = cost_fucntion.get_min_boundary()

        self.step_size = 0.1 * (self.max_boundary - self.min_boundary)

    def __single_point_crossover(self, parent1, parent2):
        cut_point = random.randint(low=1, high=self.dimension - 1)

        child1 = np.append(parent1[0:cut_point], parent2[cut_point:])
        child2 = np.append(parent2[0:cut_point], parent1[cut_point:])
        return child1, child2

    def __double_point_crossover(self, parent1, parent2):
        cut_point1 = cut_point2 = 0
        while cut_point2 <= cut_point1:
            cut_point1 = random.randint(low=1, high=self.dimension - 1)
            cut_point2 = random.randint(low=1, high=self.dimension - 1)

        child1 = np.append(parent1[0:cut_point1], parent2[cut_point1:cut_point2])
        child1 = np.append(child1, parent1[cut_point2:])

        child2 = np.append(parent2[0:cut_point1], parent1[cut_point1:cut_point2])
        child2 = np.append(child2, parent2[cut_point2:])
        return child1, child2

    def __uniform_crossover(self, parent1, parent2):
        alpha = random.randint(0, 2)  # 1 or 0, randomly.

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def crossover(self, parent1, parent2):
        random_num = random.randint(1, high=3)

        if random_num == 1:
            return self.__single_point_crossover(parent1, parent2)

        if random_num == 2:
            child1, child2 = self.__double_point_crossover(parent1, parent2)
            return child1, child2

        child1, child2 = self.__uniform_crossover(parent1, parent2)
        return child1, child2

    def mutation(self, agent, mutation_rate):

        if mutation_rate < 0 or mutation_rate > 1:
            raise Exception("Mutation rate must be between 0 and 1")

        mutation_num: int = round(mutation_rate * self.dimension)  # how many gene(input) of a chromosome should change?

        indexes = non_repeat_randint(0, self.dimension - 1, mutation_num)

        for index in indexes:
            agent[index] = agent[index] + self.step_size * random.normalvariate(mu=0, sigma=1)  # just a formula.

        return agent  # return mutated agents, the new mutated population of agents.

