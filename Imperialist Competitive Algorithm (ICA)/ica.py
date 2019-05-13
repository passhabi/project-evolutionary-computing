from module.cost_function import CostFunction
from module.randomize import permutation_int
from module.selection import RandomSelection
import numpy as np
from typing import Tuple, Union


class ICA:
    cost_func: CostFunction

    def __init__(self, cost_function: CostFunction, **kwargs):
        self.cost_func = cost_function
        self.__empires_list = list()

        self.__n_pop = kwargs['population_size']  # number of agent.
        self.__n_imp = kwargs['imperialist_size']  # number of imperialist.
        self.__n_col = self.__n_pop - self.__n_imp  # number of colonies.

        # todo: following should be removed:
        # create n_pop countries and compute cost function or FF of each one.
        self.countries = self.create_countries(self.__n_pop)
        self.costs = self.get_cost(self.countries)

        imperialists, colonies = self.select_imperialists(self.countries, self.costs)
        n_of_imps_colony_list = self.__init_colonies_of_empire(self.get_cost(imperialists))
        self.__init_create_empires(imperialists, n_of_imps_colony_list, colonies)

    def create_countries(self, size):
        countries = permutation_int([size, self.cost_func.dimensions])  # create an n * d dimensional matrix.
        return countries

    def get_cost(self, countries):
        return self.cost_func.compute_cost(countries)

    def select_imperialists(self, countries, countries_cost):
        """
        select N_imp of the most powerful countries to form the empires.
        :return:
        :param countries:
        :param countries_cost:
        :return: list of (1st) imperialist and (2nd) colonies with their costs as separate tuples.
        """
        colonies = countries.copy()
        colonies_cost = countries_cost.copy()  # no need
        # more further we take imperialist from colonies!

        sorted_cost_index = np.argsort(colonies_cost)  # dec order, lower cost = more powerful.
        # self.imp_indices = sorted_cost_index

        # select top countries base on the sorted_index:
        imperialist = []
        # imperialist_cost = []
        for idx in sorted_cost_index:
            imperialist.append(colonies[idx])
            # imperialist_cost.append(colonies_cost.pop(idx)) no need
            if len(imperialist) >= self.__n_imp:
                break

        return imperialist, colonies[10:]

    # todo: how to call this one?
    def __init_colonies_of_empire(self, imperialist_cost):

        # we define the normalized cost of an imperialist by Cn = cn - max{ i }
        # Where cn is the cost of nth imperialist and Cn is its normalized cost.
        Cn = max(imperialist_cost) - imperialist_cost  # broadcasting

        # the normalized power of each imperialist is defined by Pn = |Cn / ∑ Ci|
        sigma_cost = sum(Cn)
        Pn = abs(Cn / sigma_cost)
        # From another point of view, the normalized power of an imperialist is the portion of colonies that
        # should be possessed by that imperialist.

        # The initial number of colonies of an empire will be N.C.n = round{(Pn . n_col }
        # Where N.C.n is the initial number of colonies of nth empire and n_col is the number of all colonies.

        n_of_imps_colony = np.round(Pn * self.__n_col)  # row vector - corresponding number of colonies for each empire.
        return n_of_imps_colony

    def __init_create_empires(self, imperialist, n_of_imps_colony, colonies):

        n_of_imps_colony = [0] + n_of_imps_colony  # fix start and stop indexing in for loop

        start, stop = 0, 0
        for i in range(len(imperialist)):
            empire = Empire(self.cost_func, imperialist[i])
            # todo: are colonies shuffled?

            if n_of_imps_colony[i] == 0:
                continue

            start = stop
            stop = stop + int(n_of_imps_colony[i])

            empire.add_colony(colonies[start:stop])
            self.__empires_list.append(empire)  # save the empire

    def inter_competitive(self):
        """
        Imperialistic Competition
        This imperialistic competition gradually brings about a decrease in the power of weaker empires and
        an increase in the power of more powerful ones.
        :return:
        """

        if len(self.__empires_list) <= 1:  # if we have less than one empire than there is no competition.
            return

        powerless_empire: Empire = self.__empires_list[0]

        total_powers_list = []
        for empire in self.__empires_list:
            empire.update_total_power()
            total_powers_list.append(empire.get_total_power())
            # find the most powerless empire:
            if empire.get_total_power() > powerless_empire.get_total_power():
                powerless_empire = empire

        # now we have the most powerless empire!
        # picking up the weak colony from the powerless empire:
        colony, _ = powerless_empire.take_weak_colony()
        # find an empire to hand the colony to it:
        index = self.prob_of_possession_colony(total_powers_list)
        empire: Empire = self.__empires_list[index]
        colony = [colony]
        empire.add_colony(colony)

    @staticmethod
    def prob_of_possession_colony(total_powers_list):

        # The normalized total cost is simply obtained by N.T.Cn = T.Cn - max{ T.Ci }
        # Where T.C.n and N.T.C.n are respectively total cost and normalized total cost of nth empire.
        total_powers_list = np.array(total_powers_list)
        Tc = max(total_powers_list) - total_powers_list  # broadcasting

        # the normalized power of each imperialist is defined by Pp = |Tc / ∑ Ti|
        sigma_cost = sum(Tc)

        # To divide the mentioned colonies among empires based on the possession probability of them,
        #   we form the vector P as: P = [ Pp1 Pp2 Pp3 .. Ppn_imp ]
        P = abs(Tc / sigma_cost)

        # Then we create a vector with the same size as P whose elements are uniformly distributed random numbers.
        # R = [r1 , r2 , r3 ,..., r n_imp ]
        n_imp = len(total_powers_list)
        R = np.random.uniform(0, 1, size=[1, n_imp])

        # Then we form vector D by simply subtracting R from P:
        D = P - R
        # Referring to vector D we will hand the mentioned colonies to an empire whose relevant index in D is maximum:
        return int(np.argmax(D))

    def empire_collapse_check(self):

        colony = None
        for i in range(len(self.__empires_list)):
            if self.__empires_list[i].get_colonies_size() < 1:
                # get itself as colony:
                colony = self.__empires_list[i].collapse()
                self.__empires_list.pop(i)

        if colony is None:
            return

        # update n_imp parameter:
        self.__n_imp = len(self.__empires_list)

        # find an empire to hand the colony:
        total_powers_list = []
        for empire in self.__empires_list:
            total_powers_list.append(empire.get_total_power())

        index = self.prob_of_possession_colony(total_powers_list)
        empire: Empire = self.__empires_list[index]
        empire.add_colony(colony)

    def show_result(self):

        best_solution = np.inf
        for empire in self.__empires_list:
            imp, imp_cost = empire.get_imperialist()

            if imp_cost < best_solution:
                best_solution = imp_cost

        return best_solution

    def assimilation(self):
        for empire in self.__empires_list:
            empire.assimilation()

    def intra_competitive(self):
        for empire in self.__empires_list:
            empire.intra_competitive()


class Empire:
    cost_func = CostFunction

    def __init__(self, cost_func: CostFunction, imperialist):
        super().__init__()
        self.zetha = 0.1
        self.cost_func = cost_func
        self.__imperialist = imperialist
        self.__imperialist_cost = self.get_cost(imperialist)

        self.__colonies_list = []
        self.__colonies_cost_list = []

        self.__total_power = 0

    def get_cost(self, countries):
        return self.cost_func.compute_cost(countries)

    def get_imperialist(self):
        return self.__imperialist, self.__imperialist_cost

    def add_colony(self, colony: Union[int, list]):
        self.__colonies_list.extend(colony)
        self.__colonies_cost_list = self.get_cost(self.__colonies_list)

        # todo: self.__total_power

    def get_total_power(self):
        return self.__total_power

    def get_colonies_size(self):
        return len(self.__colonies_list)

    def assimilation(self):

        self.__colonies_cost_list = [0] * len(self.__colonies_list)

        i = 0
        for colony in self.__colonies_list:
            beta = np.random.uniform(0, 2)  # why beta < 2? more than that converge to imperialist too soon.
            # β is a number greater than 1 and d is the distance between colony and imperialist.
            colony = np.array(colony).reshape(1, self.cost_func.get_dimensions())
            # colony += beta * (self.__imperialist - colony)
            np.add(colony, beta * (self.__imperialist - colony), out=colony, casting="unsafe")
            colony = colony.tolist()
            # bounding:
            colony = self.bounding(colony)
            # get corresponding cost of the colony:
            self.__colonies_cost_list[i] = self.get_cost(colony)[0]
            i += 1

        # n_col = len(self.__colonies_list)
        # beta = np.random.uniform(0, 2, [1, n_col])
        # # β is a number greater than 1 and d is the distance between colony and imperialist.
        #
        # self.__colonies_list += beta * (self.__imperialist - self.__colonies_list)
        # # get corresponding cost of the colony:
        # self.__colonies_cost_list = self.get_cost(self.__colonies_list)

    def bounding(self, colony):

        bounded_colony = []
        for item in colony[0]:
            item = max(item, self.cost_func.min_boundary)
            item = min(item, self.cost_func.max_boundary)
            bounded_colony.append(item)
        return bounded_colony

    def intra_competitive(self):
        """
        Exchanging Positions of the Imperialist and a Colony.

        :return:
        """
        if min(self.__colonies_cost_list) < self.__imperialist_cost[0]:  # if <colony got more power than imperialist>:

            # store imperialist in temporary variable:
            temp = self.__imperialist
            temp_cost = self.__imperialist_cost

            # set the colony as imperialist:
            idx_colony: int = np.argmin(self.__colonies_cost_list)
            self.__imperialist = self.__colonies_list[idx_colony]
            self.__imperialist_cost = self.__colonies_cost_list[idx_colony]

            # save the imperialist instead of that colony:
            self.__colonies_list[idx_colony] = temp
            self.__colonies_cost_list[idx_colony] = temp_cost

    def update_total_power(self):
        """
        Total power of an empire is mainly affected by the power of imperialist country.
            But the power of the colonies are negligible.

        T.C.n = Cost(imperialistn ) + ξ mean{Cost(colonies of empiren )}
        Where T.C.n is the total cost of the nth empire.
        :return:
        """
        #  and ξ is a positive number which is considered to be less than 1.
        self.__total_power = self.__imperialist_cost + self.zetha * np.mean(self.__colonies_cost_list)
        return self.__total_power

    def take_weak_colony(self):
        weak_index: int = np.argmax(self.__colonies_cost_list)
        return self.__colonies_list.pop(weak_index), self.__colonies_cost_list.pop(weak_index)

    def collapse(self):
        return self.__imperialist, [self.__imperialist_cost]
