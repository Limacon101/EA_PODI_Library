import math

n = 5


def create_solution(t):
    s_list = []
    for i in range(n):
        xi = t.random() * 1000 - 500
        s_list.append(xi * math.sin(math.sqrt(abs(xi))))
    # res = 418.9828872724339 * n - sum(s_list)
    return s_list


def fitness(bits):
    return 418.9828872724339 * n - sum(bits)
