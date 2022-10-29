from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

bayes_network = BayesianNetwork()
bayes_network.add_node('M')
bayes_network.add_node("U")
bayes_network.add_node("R")
bayes_network.add_node("B")
bayes_network.add_node("S")

bayes_network.add_edge("M", "R")
bayes_network.add_edge("U", "R")
bayes_network.add_edge("B", "R")
bayes_network.add_edge("B", "S")
bayes_network.add_edge("R", "S")

def test_pgmpy():
    cpd_A = TabularCPD('M', 2, values=[[.95], [.05]])
    cpd_U = TabularCPD('U', 2, values=[[.85], [.15]])
    cpd_H = TabularCPD('B', 2, values=[[.90], [.10]])

    cpd_S = TabularCPD('S', 2, values=[[0.98, .88, .95, .6], [.02, .12, .05, .40]],
                    evidence=['R', 'B'], evidence_card=[2, 2])

    cpd_R = TabularCPD('R', 2,
                    values=[[0.96, .86, .94, .82, .24, .15, .10, .05], 
                            [.04, .14, .06, .18, .76, .85, .90, .95]],
                    evidence=['M', 'B', 'U'], 
                    evidence_card=[2, 2,2])

    bayes_network.add_cpds(cpd_A, cpd_U, cpd_H, cpd_S, cpd_R)
    check = bayes_network.check_model()

    solver = VariableElimination(bayes_network)

    result = solver.query(variables=['R'], evidence={'M': 1})
    print(type(result))