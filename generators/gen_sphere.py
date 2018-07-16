n = 30


def create_solution(t):
    return [t.random()*2-1 for _ in range(n)]


def fitness(bits):
    return sum([b**2 for b in bits])
