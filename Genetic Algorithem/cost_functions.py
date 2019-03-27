import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# from reproduction import non_repeat_randint


class CostFunction:
    dimensions: int
    min_boundary: int
    max_boundary: int

    def get_dimensions(self):
        return self.dimensions

    def get_max_boundary(self):
        return self.max_boundary

    def get_min_boundary(self):
        return self.min_boundary

    def compute_fitness(self, agent_row):
        raise NotImplementedError


class Sphere(CostFunction):

    def __init__(self):
        self.dimensions = 5

        # maximum and minimum of response boundary in each __dimension:
        self.min_boundary = -10
        self.max_boundary = +10

    def compute_fitness(self, agent_row):
        fitness = np.sum(agent_row ** 2, axis=1)
        return fitness


class TravellingSalesmanProblem(CostFunction):

    def __init__(self, num_of_cities: int, distance_range: int = None):
        self.dimensions = num_of_cities

        # maximum and minimum of response boundary in each __dimension:
        self.min_boundary = 0
        self.max_boundary = num_of_cities - 1

        # virtual distance between cites:
        if distance_range is not None:
            self.distance_range = distance_range
            self.__generate_cities()
        else:
            self.distance_range = None
        # plot the cities:

    def __generate_cities(self):
        x_axises = np.random.randint(0, self.distance_range, size=self.dimensions)
        y_axises = np.random.randint(0, self.distance_range, size=self.dimensions)

        # compute distance:s
        self.distance_matrix = np.zeros(
            [self.dimensions, self.dimensions])  # make a empty n×n matrix, __dimension = number of cities

        # compute euclidean distance between cities and store it in distance matrix:
        for row in range(0, self.dimensions - 1):  # this was dimensions - 1
            for column in range(row + 1, self.dimensions):  # this was row + 1
                self.distance_matrix[row, column] = np.sqrt(np.exp2(x_axises[row] - x_axises[column]) + np.exp2(
                    y_axises[row] - y_axises[column]))  # upper triangular matrix
                # diagonal is zero:
                # if row == column:
                #     self.distance_matrix[column, row] = np.inf
                self.distance_matrix[column, row] = self.distance_matrix[row, column]  # and lower triangular matrix..
                #                                                                        is the same is the upper.
        # join x and y axises, first row: x axises second is y axises:
        self.cities = np.append(x_axises.reshape(1, -1), y_axises.reshape(1, -1), axis=0)

        self.plot_cities()

    def plot_cities(self):
        plt.scatter(self.cities[0, :], self.cities[1, :], marker='o')
        plt.show()

    def plot_agent_travel_order(self, agent_row):
        agent_row = np.append(agent_row, agent_row[0])  # adding the first element to the last. e.g. 5 > 4 > 3 > [5].

        agent_row = agent_row.astype(int)

        plt.scatter(self.cities[0, :], self.cities[1, :], marker='o')
        # annotate_cities = np.arange(1, self.dimensions + 1)

        # add number annotate to cities:
        # for num in annotate_cities:
        #     aplot.annotate(num, (self.cities[0, num], self.cities[1, num]))

        plt.plot(self.cities[0, agent_row], self.cities[1, agent_row])
        plt.show()

    def create_cities(self):
        # todo: user can create its own cities
        pass

    def compute_fitness(self, agent):
        """

        :param agent: matrix is a combination order to travel to cities.
        :return: cost of the given order.
        """
        if self.distance_range is None:
            raise Exception("There are no cities; Initialize distance_range on"
                            "object definition whether use create_cities() function to make your own cities")

        cost = 0
        # adding the first element to the last. e.g. 5 > 4 > 3 > [5].
        agent = np.hstack((agent, agent[:, 0].reshape(-1, 1)))
        # need a loop to travel to the first city.

        agent = agent.astype(int)
        for index in range(0, self.dimensions):
            i = agent[:, index]  # distance of the first city)to
            ii = agent[:, index + 1]  # the second city is following:
            cost += self.distance_matrix[i, ii]

        '''
        # find duplicate gene:
        none_duplicates = []
        agent = agent[:, :-1]  # dont count last gene, due to the adding the first element with hstack.
        for row in range(agent.shape[0]):
            count = 1
            for gene in agent[row]:
                if gene not in none_duplicates:
                    none_duplicates.append(gene)
                else:
                    cost[row] = np.inf
        '''
        return cost
