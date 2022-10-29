from causallearn.search.ConstraintBased.FCI import fci
import numpy as np
from causallearn.utils.GraphUtils import GraphUtils

path = "causal-attributes-generation/"
file = open("causal-attributes-generation/user_info.csv", 'r')

data = np.loadtxt(file, delimiter=",", skiprows=1)
data = np.delete(data, 0, axis = 1)

G, edges = fci(data, independence_test_method="fisherz")

# visualization
pdy = GraphUtils.to_pydot(G) 
pdy.write_png(path+"simple_test.png")