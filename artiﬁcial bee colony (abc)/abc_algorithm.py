from cost_functions import CostFunction
from selection_method import RouletteWheelSelection
import numpy as np
import pandas as pd


class ArtificialBeeColony:

    def __init__(self, cost_function: CostFunction, swarm_size: int = 20):
        self.cost_func = cost_function
        self.swarm_size = swarm_size

        self.n_onlookers = round(swarm_size / 2)  # 50% or half of the swarm.
        self.n_employed = self.n_onlookers  # same as number of onlookers.

        self.limit = self.n_onlookers * self.cost_func.dimensions  # If a solution representing a food source
        #   is not improved by a predetermined number of trials, then that food source is abandoned by its employed bee.
        self.limits = np.array([self.limit] * self.n_employed)  # for each food source we have a limit.

        # make the initialize population (send scout bees to find the food sources):
        self.employees = np.random.uniform(low=self.cost_func.min_boundary, high=self.cost_func.max_boundary,
                                           size=[self.n_employed, self.cost_func.dimensions])
        self.employees_cost = self.cost_func.compute_cost(self.employees)  # compute cost(fitness) of the employee bees.

        # to keep track and saving the best solution, we define the following:
        index = np.argmin(self.employees_cost)  # take the index of the best cost.
        self.top_food_sources = pd.DataFrame({'index': index,
                                              'position': [self.employees[index]],
                                              'cost': self.employees_cost[index]})

    def place_employed_bees(self):
        """
        #  (a) Place the employed bees on the food sources in the memory.
        # send employed bee to food sources that scout bee found (note that we didn't define
        #   scout bees and we treat employed bee as both scout and employed bee):

        :return:
        """
        for index in range(self.n_employed):
            self.flying_to_food_source(index)

    def place_onlooker_bees(self):
        """
        # (b) Place the onlooker bees on the food sources in the memory.
        # An onlooker bee chooses a food source depending on the probability value associated with that food source,
        #   pi, calculated by the following expression (2.1):
        # pi = ﬁti / Σ ﬁtn  where ﬁti is the ﬁtness value of the solution i evaluated by its employed bee.

        :return:
        """

        # employees_cost_reverse = max(
        #     employees_cost) - employees_cost  # due minimization, we need more probability for a small cost.
        # probabilities = employees_cost_reverse / sum(employees_cost_reverse)

        # roulette_wheel:
        rws = RouletteWheelSelection()
        rws.make_roulette_wheel(self.employees_cost)

        for index in range(self.n_onlookers):  # <for each food source do:>
            self.flying_to_food_source(rws.roll())

    def send_scout_to_search(self):
        """
        # (c) Send the scouts to the search area to discovering new food sources.

        :return:
        """
        for i in range(self.n_employed):
            if self.limits[i] <= 0:  # <if it's not nectar in a such food source i, do:>
                self.limits[i] = self.limit  # reset the counter.
                self.employees[i] = np.random.uniform(self.cost_func.min_boundary, self.cost_func.max_boundary,
                                                      self.cost_func.dimensions)  # find another food source.
                self.employees_cost[i] = self.cost_func.compute_cost(self.employees[i])

    def store_best_iteration_foodsource(self):
        index = np.argmin(self.employees_cost)  # take the index of the best cost.
        self.top_food_sources = self.top_food_sources.append({'index': index,
                                                              'position': [self.employees[index]],
                                                              'cost': self.employees_cost[index]}, ignore_index=True)

    def show_results(self):
        self.cost_func.plot_cost_iteration(self.top_food_sources.cost)
        solution = self.cost_func.get_print_solution(self.top_food_sources.position, self.top_food_sources.cost)
        self.cost_func.plot_solution(solution)

    def flying_to_food_source(self, i):
        # choose a random food source j ≠ k:
        k = i
        while k == i:
            k = np.random.randint(0, self.n_employed)
        # In order to produce a candidate food position from the old one, the ABC uses the
        # following expression (2.2):
        # vij = xij + φij(xij − xkj)  where k ∈ {1, 2, . . . , n_employed} and j ∈ {1, 2, . . . , dimension}
        phi = np.random.uniform(-1, 1, [1, self.cost_func.dimensions])
        v = (self.employees[i] + phi * (self.employees[k] - self.employees[i])).flatten()
        # boundary check:
        for j in range(len(v)):  # <for each element(gene) in v, do>:
            # if element is more bigger that max boundary, set it to max boundary:
            v[j] = min(v[j], self.cost_func.max_boundary)
            # if element is less that min boundary, set it to min boundary:
            v[j] = max(v[j], self.cost_func.min_boundary)

        # compute ith bee cost:
        v_cost = self.cost_func.compute_cost(v)
        if v_cost < self.employees_cost[i]:  # <if v has better cost do:>
            # replace it with ith employee:
            self.employees[i] = v
            self.employees_cost[i] = v_cost
        else:
            self.limits[i] = self.limits[i] - 1  # give a negative feedback.
