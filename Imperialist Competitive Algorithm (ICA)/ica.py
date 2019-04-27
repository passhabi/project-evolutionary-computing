from module.cost_function import CostFunction
from random import randint

# initialization:

# compute cost for population:
# choice imperialists:
# assign colonies to imperialists:
# assimilation
# intra competitive:
# find the less powerful imperialist:
# inter competitive:


class ICA:

    def __init__(self, cost_function: CostFunction, **kwargs):
        self.__cost_func = cost_function

        self.__n_pop = kwargs['population_size']  # number of agent.
        self.__n_imp = kwargs['imperialist_size']  # number of imperialist.
        self.__n_col = self.__n_pop - self.__n_imp  # number of colonies.

        # todo: initial countries:

        # todo: compute cost of countries

    def __create_countries(self):
        self.countries = [[] * self.__cost_func.get_dimensions()] * self.__n_pop
