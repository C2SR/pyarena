import numpy as np
from pyarena.world.graph_world import GraphWorld
from pyarena.plots.graph_planning import GraphPlanning
from pyarena.planning.ipp_bnb import IPPBnB


# world
size = np.array([4,4])
kwargsWorld = {'size': size}
mworld = GraphWorld(**kwargsWorld)

# Plot
kwargsPlot = {'world': mworld}
mplot = GraphPlanning(**kwargsPlot)

# Planning
kwargsPlanning = {'world': mworld}
mplanning = IPPBnB(**kwargsPlanning)

# run
start = np.array([0,0]).reshape(2,1)
goal = np.array([4,4]).reshape(2,1)
print('Starting BnB IPP')
optimal_path, optimal_cost = mplanning.run(start, goal, budget=8)
print('Find an optimal path!')
print('Objective value:', optimal_cost)

# Plot
mplot.update(optimal_path)