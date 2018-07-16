import math

n = 10


def create_solution(t):
    s_list = []
    for _ in range(n):
        xi = t.random() * 6.14 - 5.12
        s_list.append(xi ** 2 - 10 * math.cos(2 * math.pi * xi))
    return 10 * n + sum(s_list)


def fitness(sol):
    return sol
