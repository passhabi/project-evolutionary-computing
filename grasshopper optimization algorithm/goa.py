import numpy as np
from cost_functions import *


class GrasshopperOptimization:
    def __init__(self, cost_func: CostFunction = Sphere(), swarm_size=50, intensity_attraction=0.5,
                 attractive_scale=1.5):

        self.cost_func = cost_func
        self.swarm_size = swarm_size

        self.f = intensity_attraction
        self.l = attractive_scale

        # make the initial the population of grasshoppers in the response space:
        self.swarm = np.random.uniform(cost_func.lower_bound, cost_func.upper_bound,
                                       size=[swarm_size, cost_func.dimensions])
        self.swarm_costs = cost_func.compute_cost(self.swarm)

        # save the best results:
        self.position_history = []
        self.cost_history = []

        i = np.argmin(self.swarm_costs)
        self.position_history.append(self.swarm[i].copy())
        self.cost_history.append(self.swarm_costs[i])

        self.target_position = self.swarm[i]
        self.target_cost = self.swarm_costs[i]

    def distance(self, a, b):
        #              ___________________________________________________
        # d(a, b) =  √(a(1)-b(1))^2 + (a(2)-b(2))^2 + ... + (a(n)-b(n))^2
        d = (a - b) ** 2  # (subtracting element by element) ^ 2
        return np.sqrt(sum(d))

    def s_func(self, r):
        return (self.f * np.exp(-r / self.l)) - np.exp(-r)  # Eq.(2.3) in the paper

    @staticmethod
    def compute_c(epoch, max_iteration):
        # update c using eq. 2.8:
        c_max = 1
        c_min = 0.00001
        return c_max - epoch * ((c_max - c_min) / max_iteration)

    def change_grasshoppers_position(self, c):
        for i in range(self.swarm_size):
            sigma = 0
            for j in range(self.swarm_size):
                if i != j:
                    distance_ij = self.distance(self.swarm[i], self.swarm[j])  # Calculate the distance between two grasshoppers
                    distance_ij_hat = (self.swarm[j] - self.swarm[i]) / (distance_ij + 1e-20)  # xj - xi / dij in Eq.(2.7)
                    xj_xi = 2 + (distance_ij % 2)  # | xjd - xid | [1-4]
                    xj_xi = self.s_func(xj_xi)  # s(| xjd - xid |) in Eq.(2.7)

                    # The first part inside the big bracket in Eq. (2.7):
                    sigma += ((self.cost_func.upper_bound - self.cost_func.lower_bound) * c / 2) * xj_xi * distance_ij_hat
            # update the grasshopper ith position:
            self.swarm[i] = c * sigma + self.target_position  # Eq. (2.7) in the paper

            # boundary check:
            for j in range(self.cost_func.dimensions):  # <for each element(gene) in v, do>:
                # if element is more bigger that max boundary, set it to max boundary:
                self.swarm[i, j] = min(self.swarm[i, j], self.cost_func.upper_bound)
                # if element is less that min boundary, set it to min boundary:
                self.swarm[i, j] = max(self.swarm[i, j], self.cost_func.lower_bound)


    def update_cost_and_target(self):
        self.swarm_costs = self.cost_func.compute_cost(self.swarm)
        i = np.argmin(self.swarm_costs)

        if self.target_cost > self.swarm_costs[i]:  # if <swarm_cost was better>:
            self.target_position = self.swarm[i].copy()
            self.target_cost = self.swarm_costs[i].copy()

        # save the best results of the iteration:
        self.position_history.append(self.target_position.copy())
        self.cost_history.append(self.target_cost.copy())


    def show_step_result(self, epoch):
        self.cost_func.print_step_result(self.position_history[-1], epoch)
        # self.cost_func.visual_result(self.position_history[-1])

    def show_final_result(self):
        self.cost_func.plot_cost_vs_iteration(self.cost_history)

        i = np.argmin(self.cost_history)
        print("The best result details:")
        self.cost_func.print_step_result(self.position_history[i], i)
        self.cost_func.visual_result(self.position_history[i])