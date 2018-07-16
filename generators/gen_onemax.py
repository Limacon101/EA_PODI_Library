n = 50


def create_solution(t):
    """
    Creates a solution to the Onemax problem

    :param t:       A Trace object that is used to generate random numbers
    :return:        A random solution
    """
    return [t.randint(0, 1) for _ in range(n)]


def fitness(bits):
    """
    Gets a numeric value corresponding to the fitness of a given solution

    :param bits:    A solution
    :return:        A fitness value
    """
    return sum(bits)
