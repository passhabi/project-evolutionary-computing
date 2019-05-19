import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


class CostFunction:

    def __init__(self, dimensions, min_boundary, max_boundary, name):
        self.__dimensions = dimensions
        # maximum and minimum of response boundary in each dimension:
        self.__min_boundary = min_boundary
        self.__max_boundary = max_boundary
        self.__costFunction_name = name  # only a name for plotting

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def min_boundary(self):
        return self.__min_boundary

    @property
    def max_boundary(self):
        return self.__max_boundary

    @staticmethod
    def get_print_solution(agents_rows, costs):
        """
        Print the founded solution so far and return it.
        :param agents_rows: chromosome list.
        :param costs: fitness list.
        :return: an agent_row (chromosome).
        """
        # show the best founded solution:
        index = np.argmin(costs)
        print("best founded solution is:")
        print("position: ", agents_rows[index])
        print("with the cost: ", costs[index])
        return agents_rows[index]

    def plot_solution(self, solution):
        """

        :param solution: solution got by get_print_solution method. its a agent_row or (chromosome)
        :return:
        """
        raise NotImplementedError

    def plot_cost_iteration(self, costs):
        plt.plot(costs)
        plt.title(self.__costFunction_name)
        plt.xlabel('iteration')
        plt.ylabel('Cost')
        plt.show()

    def compute_cost(self, agents_rows):
        raise NotImplementedError


class Sphere(CostFunction):

    def __init__(self):
        # set Sphere parameters:
        super().__init__(dimensions=5, min_boundary=-10, max_boundary=+10, name="Sphere")

    def compute_cost(self, agents_rows):
        agents_rows = agents_rows.reshape(-1, self.dimensions)
        cost = np.sum(agents_rows ** 2, axis=1)
        return cost

    def plot_solution(self, solution):
        raise Exception("there is not plot for Sphere problem.")


class NQueen(CostFunction):

    def __init__(self, num_of_queen: int = 8):
        super().__init__(dimensions=num_of_queen, min_boundary=0, max_boundary=1, name="N Queen")

    def get_print_solution(self, agents_rows, costs):
        # change coding representation to discrete number:
        # instead of decoding all the rows in agents_rows we only pass the the best one,
        #   to take a weight off print_solution work.
        index = np.argmin(costs)  # find the best agent's index.

        agents_row = np.array(agents_rows[index]).flatten()
        agents_row = np.argsort(agents_row)  # change the representation (decoding to discrete numbers).

        cost = costs[index]

        print("best founded solution is:")
        print("position: ", agents_row)
        print("with the cost: ", cost)
        return agents_row

    def plot_solution(self, solution: list):
        """
        Show a representation of N Queen problem with the given solution.
        :param solution: solution got by get_print_solution method. its a agent_row or (chromosome)
        :return:
        """
        size = len(solution)
        for row in range(size):
            line = "  "
            for col in range(size):
                if solution[row] == col:
                    line += "👑 "
                else:
                    line += "⬜ "
            print(line)

    def compute_cost(self, agents_rows):

        if len(np.shape(agents_rows)) == 1:  # < if there is only one row in agents_rows, do>:
            agent = np.argsort(agents_rows)  # change coding representation to discrete number.
            return self.compute_cost_of_1_agent(agent)

        costs = []
        for agent in agents_rows:
            # add computed fitness to the list of costs:
            agent = np.argsort(agent)  # change coding representation to discrete number.
            costs = np.append(costs, self.compute_cost_of_1_agent(agent))
        return costs

    @staticmethod
    def compute_cost_of_1_agent(chromosome):
        # compute cost for an agent or chromosome
        cost = 0
        columns = [i + 1 for i in range(len(chromosome))]
        rows = [j + 1 for j in range(len(chromosome))]

        for col in columns:
            for row in rows:
                if col != row:
                    # if there is a hit:
                    if chromosome[col - 1] == chromosome[row - 1]:
                        cost += 1
                    if abs(col - row) == abs(chromosome[col - 1] - chromosome[row - 1]):
                        cost += 1
        return cost


class TravellingSalesmanProblem(CostFunction):

    def __init__(self, num_of_cities: int = None, distance_range: int = None):
        super().__init__(num_of_cities, min_boundary=0, max_boundary=1, name="TSP")

        # virtual distance between cites:
        self.x_axises = None
        self.y_axises = None

        if distance_range is not None:
            self.distance_range = distance_range
            self.__generate_cities()
        else:
            self.distance_range = None

    def __generate_cities(self):
        self.x_axises = np.random.randint(0, self.distance_range, size=self.dimensions)
        self.y_axises = np.random.randint(0, self.distance_range, size=self.dimensions)

        self.__compute_distance()
        # self.plot_cities()

    def __compute_distance(self):
        # compute distance:s
        self.distance_matrix = np.zeros(
            [self.dimensions, self.dimensions])  # make a empty n×n matrix, __dimension = number of cities

        # compute euclidean distance between cities and store it in distance matrix:
        for row in range(0, self.dimensions - 1):  # this was dimensions - 1
            for column in range(row + 1, self.dimensions):  # this was row + 1
                self.distance_matrix[row, column] = np.sqrt(
                    np.exp2(self.x_axises[row] - self.x_axises[column]) + np.exp2(
                        self.y_axises[row] - self.y_axises[column]))  # upper triangular matrix
                # diagonal is zero:
                # if row == column:
                #     self.distance_matrix[column, row] = np.inf
                self.distance_matrix[column, row] = self.distance_matrix[row, column]  # and lower triangular matrix..
                #                                                                        is the same is the upper.
        # join x and y axises, first row: x axises second is y axises:
        self.cities = np.append(self.x_axises.reshape(1, -1), self.y_axises.reshape(1, -1), axis=0)

    def plot_cities(self):
        plt.scatter(self.cities[0, :], self.cities[1, :], marker='o')
        plt.show()

    def get_print_solution(self, agents_rows, costs):
        """

        :param agents_rows:
        :param costs:
        :return: The found solution so far.
        """
        # change coding representation to discrete number:
        # instead of decoding all the rows in agents_rows we only pass the the best one,
        #   to take a weight off print_solution work.
        index = np.argmin(costs)  # find the best agent's index.

        agents_row = np.array(agents_rows[index]).flatten()
        agents_row = np.argsort(agents_row)  # change the representation (decoding to discrete numbers).

        cost = costs[index]

        print("best founded solution is:")
        print("position: ", agents_row)
        print("with the cost: ", cost)
        return agents_row

    def plot_solution(self, solution):
        """
        Show a representation of TSP problem with the given solution.
        :param solution: solution got by get_print_solution method. its a agent_row or (chromosome)
        :return:
        """
        solution = np.append(solution, solution[0])  # adding the first element to the last. e.g. 5 > 4 > 3 > [5].

        solution = solution.astype(int)

        plt.scatter(self.cities[0, :], self.cities[1, :], marker='o')
        # annotate_cities = np.arange(1, self.dimensions + 1)

        # add number annotate to cities:
        # for num in annotate_cities:
        #     aplot.annotate(num, (self.cities[0, num], self.cities[1, num]))

        plt.plot(self.cities[0, solution], self.cities[1, solution])
        plt.show()

    def compute_cost(self, agents_rows):
        """

        :param agents_rows: matrix is a combination order to travel to cities.
        :return: cost of the given order.
        """
        # change coding representation to discrete number:

        if self.distance_range is None:
            raise Exception("There are no cities; Initialize distance_range on "
                            "object definition whether use create_cities() function to make your own cities.")
        cost = 0

        one_agents = len(agents_rows.shape) == 1  # do we have just one agent? (agents_rows contain only one row?)

        # adding the first element to the last. e.g. 5 > 4 > 3 > [5].
        if one_agents:
            agents_rows = np.argsort(agents_rows)
            solution = np.append(agents_rows, agents_rows[0])

            # need a loop to travel to the all cities:
            solution = solution.astype(int)  # todo: remove this
            for index in range(0, self.dimensions):
                i = solution[index]  # distance of the first city to
                ii = solution[index + 1]  # the second city is following:
                cost += self.distance_matrix[i, ii]
        else:
            agents_rows = np.argsort(agents_rows, axis=1)
            solution = np.hstack((agents_rows, agents_rows[:, 0].reshape(-1, 1)))

            # need a loop to travel to the all cities:
            solution = solution.astype(int)  # todo: remove this
            for index in range(0, self.dimensions):
                i = solution[:, index]  # distance of the first city to
                ii = solution[:, index + 1]  # the second city is following:
                cost += self.distance_matrix[i, ii]

        return cost

"""
    def create_cities(self, x: list, y: list):
        create cities by user.
        :param x: x vector axis parameters
        :param y: y vector axis parameters
        :return:

        if len(x) != len(y):
            raise Exception("x and y must be same size")
        self.x_axises = np.array(x)
        self.y_axises = np.array(y)

        num_of_cities = len(x)
        self.dimensions = num_of_cities

        # maximum and minimum of response boundary in each __dimension:
        self.min_boundary = 0
        self.max_boundary = num_of_cities - 1

        self.distance_range = True

        self.__compute_distance()
"""
