import matplotlib.pyplot as plt
import numpy as np


class CostFunction:
    """
    note:
    we call a chromosome, solution vector.
    """

    def __init__(self, dimensions, lower_bound, upper_bound, name):
        self.__dimensions = dimensions
        # maximum and minimum of response boundary in each dimension:
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__costFunction_name = name  # only a name for plotting

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def lower_bound(self):
        return self.__lower_bound

    @property
    def upper_bound(self):
        return self.__upper_bound

    @staticmethod
    def _discrete_decoding(original_function):
        """
        Decorator function that decode continues space to discrete space by sorting.
        for example consider given solution_vector [5.1, 6.3, 4.008, 9.1, 6.02] to the original function.

        This decorator make the original function to work with discrete version of the given solution_vector which in this
        example is [2 0 4 1 3].
        This can be used in discrete problems like N-Queen or TSP.
        """

        def wrapper(self, solution_vectors: np.ndarray, *args, **kwargs):
            if len(solution_vectors.shape) > 1:
                solution_vectors = np.argsort(solution_vectors, axis=1).reshape(-1, self.dimensions)
            else:
                solution_vectors = np.argsort(solution_vectors).reshape(1, self.dimensions).flatten()
            return original_function(self, solution_vectors, *args, **kwargs)
        return wrapper

    def compute_cost(self, solution_vectors):
        """
        temporary docstring
        :param solution_vectors:
        :return:
        """
        raise NotImplementedError

    def visual_result(self, solution_vector):
        raise NotImplementedError

    def plot_cost_vs_iteration(self, costs):
        plt.plot(costs)
        plt.title(self.__costFunction_name)
        plt.xlabel('iteration')
        plt.ylabel('Cost')
        plt.show()

    def print_step_result(self, solution_vector, iteration: int = ""):
        """
        Use this to print result in each iteration.
        This prints a simple result of the last solution (an chromosome) that its cost has been computed.
        """
        cost = self.compute_cost(solution_vector)
        print(f"solution {solution_vector} with the cost: {cost} in the iteration {iteration}")


class Sphere(CostFunction):

    def __init__(self):
        # set Sphere parameters:
        super().__init__(dimensions=5, lower_bound=-10, upper_bound=+10, name="Sphere")

    def compute_cost(self, solution_vectors: np.ndarray):
        solution_vectors = np.array(solution_vectors)
        solution_vectors = solution_vectors.reshape(-1, self.dimensions)
        cost = np.sum(solution_vectors ** 2, axis=1)
        return cost

    def visual_result(self, solution_vector):
        pass


class NQueen(CostFunction):

    def __init__(self, num_of_queen: int = 8):
        super().__init__(dimensions=num_of_queen, lower_bound=0, upper_bound=1, name="N Queen")

    @CostFunction._discrete_decoding
    def visual_result(self, solution_vector):
        """
        Show a representation of N Queen problem with the given solution.
        :param solution_vector: solution got by get_print_solution method. its a agent_row or (chromosome)
        :return:
        """
        size = len(solution_vector)
        for row in range(size):
            line = "  "
            for col in range(size):
                if solution_vector[row] == col:
                    line += "👑 "
                else:
                    line += "⬜ "
            print(line)

    @CostFunction._discrete_decoding
    def compute_cost(self, solution_vectors):

        if len(np.shape(solution_vectors)) == 1:  # < if there is only one row in agents_rows, do>:
            return self.__compute_cost_of_a_row(solution_vectors)

        costs = []
        for agent in solution_vectors:
            # add computed fitness to the list of costs:
            costs = np.append(costs, self.__compute_cost_of_a_row(agent))
        return costs

    @CostFunction._discrete_decoding
    def print_step_result(self, solution_vector, iteration: int = ""):
        super().print_step_result(solution_vector, iteration)

    def __compute_cost_of_a_row(self, agent_row):
        # compute cost for an agent or chromosome

        x = list(range(self.dimensions))
        y = agent_row
        # y = np.argsort(agent_row)  # change coding representation to discrete number. #todo: did you replaced?

        cost = 0
        for i in range(self.dimensions - 1):
            for j in range(i + 1, self.dimensions):
                if np.abs(x[i] - x[j]) == np.abs(y[i] - y[j]):
                    cost = cost + 1
        return cost


class TravellingSalesmanProblem(CostFunction):

    def __init__(self, num_of_cities: int = None, distance_range: int = None):
        super().__init__(num_of_cities, lower_bound=0, upper_bound=1, name="TSP")

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

    @CostFunction._discrete_decoding
    def compute_cost(self, solution_vectors):
        """

        :param solution_vectors: matrix is a combination order to travel to cities.
        :return: cost of the given order.
        """
        # change coding representation to discrete number:

        if self.distance_range is None:
            raise Exception("There are no cities; Initialize distance_range on "
                            "object definition whether use create_cities() function to make your own cities.")
        cost = 0

        one_agents = len(solution_vectors.shape) == 1  # do we have just one agent? (agents_rows contain only one row?)

        # adding the first element to the last. e.g. 5 > 4 > 3 > [5].
        if one_agents:
            # solution_vectors = np.argsort(solution_vectors)
            solution = np.append(solution_vectors, solution_vectors[0])

            # need a loop to travel to the all cities:
            solution = solution.astype(int)  # todo: remove this
            for index in range(0, self.dimensions):
                i = solution[index]  # distance of the first city to
                ii = solution[index + 1]  # the second city is following:
                cost += self.distance_matrix[i, ii]
        else:
            # solution_vectors = np.argsort(solution_vectors, axis=1)
            solution = np.hstack((solution_vectors, solution_vectors[:, 0].reshape(-1, 1)))

            # need a loop to travel to the all cities:
            solution = solution.astype(int)  # todo: remove this
            for index in range(0, self.dimensions):
                i = solution[:, index]  # distance of the first city to
                ii = solution[:, index + 1]  # the second city is following:
                cost += self.distance_matrix[i, ii]

        return cost

    @CostFunction._discrete_decoding
    def visual_result(self, solution_vector):
        """
        Show a representation of TSP problem with the given solution.
        :param solution_vector: solution got by get_print_solution method. its a agent_row or (chromosome)
        """
        # adding the first element to the last. e.g. 5 > 4 > 3 > [5]:
        solution = np.append(solution_vector, solution_vector[0])

        solution = solution.astype(int)

        plt.scatter(self.cities[0, :], self.cities[1, :], marker='o')
        # annotate_cities = np.arange(1, self.dimensions + 1)

        # add number annotate to cities:
        # for num in annotate_cities:
        #     aplot.annotate(num, (self.cities[0, num], self.cities[1, num]))

        plt.plot(self.cities[0, solution], self.cities[1, solution])
        plt.show()

    @CostFunction._discrete_decoding
    def print_step_result(self, solution_vector, iteration: int = ""):
        super().print_step_result(solution_vector, iteration)


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
