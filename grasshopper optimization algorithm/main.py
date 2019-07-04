# 29 May 2019 19:58

from goa import GrasshopperOptimization
from cost_functions import *

max_iteration = 100
goa = GrasshopperOptimization()

for epoch in range(max_iteration):
    # update c using eq. 2.8:
    c = goa.compute_c(epoch, max_iteration)

    goa.change_grasshoppers_position(c)

    goa.update_cost_and_target()

    goa.show_step_result(epoch)

print("------------------------")
goa.show_final_result()