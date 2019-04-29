from ica import ICA
from module.cost_function import *

# initialization:
tsp = TravellingSalesmanProblem(num_of_cities=8, distance_range=100)
# compute cost for population:
# choice imperialists:
# assign colonies to imperialists:
ica = ICA(tsp, population_size=50, imperialist_size=10)

for i in range(2):
    ica.assimilation()
    ica.intra_competitive()
    ica.inter_competitive()
    ica.empire_collapse_check()


