import sys
import parser
import math
from core import ops, solver

from algorithms import algo_ea

n = 25
max_nodes = 10
operators = ['+', '-', '*']
constant_range = [0, 5]

# Don't change input_var
input_var = 'X'
# input_func = lambda x: x**2 + 2
# input_func = lambda x: x**3 + x*x + 2*x - 5
input_func = lambda x: x**4 + x*(x+2) - 5
input_range = [-5, 5]
zero_div_penalty = 1000


class BNode:
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

    @staticmethod
    def print_nodes_in_order(root):
        s = []
        q = []
        q.append(root)

        while len(q) > 0:
            # Dequeue node and make it root
            root = q.pop(0)
            s.append(root)

            # Enqueue right
            if root.right:
                q.append(root.right)
            # Enqueue left
            if root.left:
                q.append(root.left)

        # Pop all items from stack:
        while len(s) > 0:
            node = s.pop()
            print(node.value)


def create_solution(t=None):
    root = BNode(random_from_list(operators, t))
    # Contains all nodes that have empty leaves (left, right = None)
    has_leaves = [root]

    # Populates tree with operators, constants and inputs
    for i in range(max_nodes // 2):
        if len(has_leaves) == 0:
            break
        node = has_leaves.pop(t.randrange(0, len(has_leaves)))
        node.left = new_node = choose_node_type(3, 2, 1, t)
        if type(new_node) == BNode:
            has_leaves.append(node.left)

        node.right = new_node = choose_node_type(3, 2, 1, t)
        if type(new_node) == BNode:
            has_leaves.append(node.right)

    # For any remaining leaves that are operators, complete with constants and inputs
    for node in has_leaves:
        node.left = choose_node_type(0, 1, 1, t)
        node.right = choose_node_type(0, 1, 1, t)

    # BNode.print_nodes(root)
    # print(get_formula(root))
    return root


def choose_node_type(w_operator, w_constant, w_input, t):
    w_sum = w_operator + w_constant + w_input

    rb = t.random()
    # print('Chose:', rb)

    r = rb * w_sum
    # r = random.uniform(0, w_sum)
    if r < w_operator:
        return BNode(random_from_list(operators, t))
    elif r < w_operator + w_constant:
        return random_constant(t)
    else:
        return input_var


# Replacement for sample for the time being
def random_from_list(l, t):
    rb = t.randrange(0, len(l))
    # print('Chose from List:', rb)
    t_ind = rb
    return l[t_ind]


def random_constant(t):
    rb = t.randint(constant_range[0], constant_range[1])
    # print('chose constant:', rb)
    return rb


def fitness(root):
    formula = get_formula(root)
    total_deviation = 0

    # print("formula:", formula)
    code = parser.expr(formula).compile()
    penalty = 0
    for i in range(input_range[0], input_range[1]):
        X = i
        # print(X, "From my formula:", eval(code))
        # print(X, "From given form:", input_func(X))

        try:
            total_deviation += abs(eval(code) - input_func(i))  # Experiment with rms
        except ZeroDivisionError:
            total_deviation += 0.001  # Prevent total_deviation finishing at 0
            penalty += zero_div_penalty
            # +1 used to prevent numbers range(0..1)
    # total_deviation = math.sqrt(total_deviation / (input_range[1] - input_range[0]))
    return total_deviation + penalty


def fitness_maximising(root):
    formula = get_formula(root)
    total_deviation = 0

    # print("formula:", formula)
    code = parser.expr(formula).compile()
    penalty = 0
    for i in range(input_range[0], input_range[1]):
        X = i
        # print(X, "From my formula:", eval(code))
        # print(X, "From given form:", input_func(X))

        try:
            total_deviation += abs(eval(code) - input_func(X)) ** 2  # Experiment with rms
        except ZeroDivisionError:
            # print('DIVISION BY ZERO!!!', get_formula(root))
            total_deviation += 1  # Prevent total_deviation finishing at 0
            penalty += zero_div_penalty
            # +1 used to prevent numbers range(0..1)
    total_deviation = math.sqrt(total_deviation / (input_range[1] - input_range[0])) + 1
    if total_deviation == 0:
        return 100
    else:
        return 100 / total_deviation - penalty


def get_formula(root):
    if type(root) is not BNode:
        return str(root)
    else:
        return "(" + get_formula(root.left) + str(root.value) + get_formula(root.right) + ")"


if __name__ == '__main__':
    # -- Show GP solutions --
    print("Genetic Programming")

    gen = sys.modules[__name__]
    alg = algo_ea
    alg_params = solver.AlgoParams(select=ops.select_tournament,
                                   crossover=ops.crossover_one_point,
                                   mutate=ops.mutate_trace_gauss,
                                   generations=500,
                                   pop_size=30,
                                   mutation_rate=0.1,
                                   minimising=True)
    s = solver.Solver(gen, alg, alg_params).solve()
    print(s)
    print(get_formula(s))
