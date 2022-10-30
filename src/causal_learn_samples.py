from causallearn.search.ConstraintBased.FCI import fci
import numpy as np
from causallearn.utils.GraphUtils import GraphUtils

def causal_learn_sample():
    file = open("user_info.csv", 'r')

    data = np.loadtxt(file, delimiter=",", skiprows=1)
    data = np.delete(data, 0, axis = 1)

    G, edges = fci(data, independence_test_method="fisherz")

    # visualization
    pdy = GraphUtils.to_pydot(G) 
    pdy.write_png("fci_causal_graph.png")
 