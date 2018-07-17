"""
Problem generator that creates random solutions for a Genetic Programming
problem -- symbolic regression. Briefly, this problem involves estimating
the (polynomial) line that best fits a set of data points.

For testing purposes, these data points are generated from a test function,
but (random) data points could be used. Solutions are represented using a
binary tree data structure.

Attributes:
    max_nodes (int): Maximum number of leaf nodes

    operators ([string]): Mathematical operators used to estimate the formula

    constant_range ([int]): Range of constants used in estimating the formula

    input_var (char): Input variable symbol (Recommended not to change)

    input_func (function): Function used to get data points

    input_range ([int]): Range of input data points (x-axis)

    zero_div_penalty (float): Solution penalty for division by zero error


"""

import sys
import parser
import math
from core import ops, solver
from algorithms import algo_ea

max_nodes = 10
operators = ['+', '-', '*']
constant_range = [0, 5]
input_var = 'X'
# input_func = lambda x: x**2 + 2
# input_func = lambda x: x**3 + x*x + 2*x - 5
input_func = lambda x: x**4 + x*(x+2) - 5
input_range = [-5, 5]
zero_div_penalty = 1000


class BNode:
    """
    Simple binary data structure used to (recursively) represent a GP solution
    """
    def __init__(self, val, left=None, right=None):
        self.value = val
        self.left = left
        self.right = right

    def __str__(self):
        return get_formula(self)

    @staticmethod
    def print_nodes(root, level=0):
        print(level, root.value)
        if type(root.left) == BNode:
            BNode.print_nodes(root.left, level + 1)
        else:
            print(level + 1, root.left)

        if type(root.right) == BNode:
            BNode.print_nodes(root.right, level + 1)
        else:
            print(level + 1, root.right)


def create_solution(t=None):
    """
    Creates a (pseudo) random GP solution.

    :param t:       Trace object
    :return:        Solution
    """
    root = BNode(_random_from_list(operators, t))
    # Contains all nodes that have empty leaves (left, right = None)
    has_leaves = [root]

    # Populates tree with operators, constants and inputs
    for i in range(max_nodes // 2):
        if len(has_leaves) == 0:
            break
        node = has_leaves.pop(t.randrange(0, len(has_leaves)))
        node.left = new_node = _choose_node_type(3, 2, 1, t)
        if type(new_node) == BNode:
            has_leaves.append(node.left)

        node.right = new_node = _choose_node_type(3, 2, 1, t)
        if type(new_node) == BNode:
            has_leaves.append(node.right)

    # For any remaining leaves that are operators, complete with constants and inputs
    for node in has_leaves:
        node.left = _choose_node_type(0, 1, 1, t)
        node.right = _choose_node_type(0, 1, 1, t)

    # BNode.print_nodes(root)
    # print(get_formula(root))
    return root


def _choose_node_type(w_operator, w_constant, w_input, t):
    """
    Choose a random node (from operators, constants and input variables)

    :param w_operator:          Weighting of choosing an operator
    :param w_constant:          Weighting of choosing a constant
    :param w_input:             Weighting of choosing an input
    :param t:                   Trace object
    :return:                    An operator, constant or input variable
    """
    w_sum = w_operator + w_constant + w_input

    rb = t.random()
    # print('Chose:', rb)

    r = rb * w_sum
    # r = random.uniform(0, w_sum)
    if r < w_operator:
        return BNode(_random_from_list(operators, t))
    elif r < w_operator + w_constant:
        return _random_constant(t)
    else:
        return input_var


# Replacement for random.sample for the time being
def _random_from_list(l, t):
    rb = t.randrange(0, len(l))
    # print('Chose from List:', rb)
    t_ind = rb
    return l[t_ind]


def _random_constant(t):
    rb = t.randint(constant_range[0], constant_range[1])
    # print('chose constant:', rb)
    return rb


def fitness(root):
    """
    Get the fitness of a GP solution. Calculated by summing the differences
    between the data points and the solution's corresponding outputs over
    the input range.

    :param root:                A solution (A BNode tree)
    :return:                    Fitness value
    """
    formula = get_formula(root)
    total_deviation = 0

    code = parser.expr(formula).compile()
    penalty = 0
    for i in range(input_range[0], input_range[1]):
        X = i
        try:
            total_deviation += abs(eval(code) - input_func(i))  # Experiment with rms
        except ZeroDivisionError:
            total_deviation += 0.001  # Prevent total_deviation finishing at 0
            penalty += zero_div_penalty
    # total_deviation = math.sqrt(total_deviation / (input_range[1] - input_range[0]))
    return total_deviation + penalty


def get_formula(root):
    """
    Returns the solution in a string format
    """
    if type(root) is not BNode:
        return str(root)
    else:
        return "(" + get_formula(root.left) + str(root.value) + get_formula(root.right) + ")"


if __name__ == '__main__':
    # Optimise GP problem using data points from the input_func lambda function
    # using an evolutionary algorithm
    print("Genetic Programming -- Symbolic Regression")
    print("\nInput data points:")
    for i in range(input_range[0], input_range[1]):
        print("(", i, ",", input_func(i), ")", end=' ', sep='')
    print()

    # Run optimisation process:
    gen = sys.modules[__name__]
    alg = algo_ea
    alg_params = solver.AlgoParams(select=ops.select_tournament,
                                   crossover=ops.crossover_one_point,
                                   mutate=ops.mutate_trace_gauss,
                                   generations=50,
                                   pop_size=30,
                                   mutation_rate=0.1,
                                   minimising=True)
    s = solver.Solver(gen, alg, alg_params).solve()
    print("\nOptimisation result and fitness:")
    print(s)
