import random
import copy
import operator
import math
from scipy import stats


def get_trunc_norm(mean, sd, lower=1e-12, upper=1):
    """
    Gets sample from truncated normal distribution

    :param mean:            Distribution mean
    :param sd:              Distribution standard deviation
    :param lower:           Lower bound
    :param upper:           Upper bound
    :return:                Real number
    """
    mean = max(lower, min(mean, upper))
    assert sd >= 0
    if sd <= 0:
        sd = 1e-12
    tn = stats.truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd).rvs()
    return max(lower, min(tn, upper))


# SELECTION
def select_random(solver, pop):
    return random.choice(pop)


def select_roulette(solver, pop, bias=1, minimising=False):
    """
    Roulette-wheel (Proportional) solution selection.

    :param solver:          Solver used to start the optimisation
    :param pop:             Population (List of Traces)
    :param bias:            Bias (higher value makes selecting better solutions more likely)
    :param minimising:      Whether the optimisation is a minimisation
    :return:                Trace object
    """
    assert min(c.fitness for c in pop) >= 0
    assert len(pop) > 0
    assert minimising is False  # Done to ensure consistency. Check could be moved to solver (roulette can only do maxi)

    point = random.uniform(0, sum(c.fitness**bias for c in pop))
    # print('Chose ', point, ' out of ', sum(c.fitness**bias for c in pop))
    for p in pop:
        point -= p.fitness**bias
        if point <= 0:
            break
    return p


def select_tournament(solver, pop, t_size=2, minimising=None):
    """
    Tournament solution selection

    :param solver:          Solver used to start the optimisation
    :param pop:             Population (List of Traces)
    :param t_size:          Tournament size
    :param minimising:      Whether the optimisation is a minimisation
    :return:                Trace object
    """
    # minimising=None: allows use of function independent of alg_params (e.g. special cases or independent testing)
    assert len(pop) >= t_size
    if minimising is None:
        minimising = solver.alg_params.minimising

    best = None
    for t in range(t_size):
        ind = random.choice(pop)
        # print('p:', ind.solution, ind.fitness)
        if minimising:
            if best is None or ind.fitness <= best.fitness:
                best = ind
        else:
            if best is None or ind.fitness >= best.fitness:
                best = ind
    return best


def select_best_n(solver, pop, n, minimising=None):
    """
    Select best n solutions from a population

    :param solver:          Solver used to start the optimisation
    :param pop:             Population (List of Traces)
    :param n:               Number of solutions
    :param minimising:      Whether the optimisation is a minimisation
    :return:
    """
    assert n <= len(pop)
    if minimising is None:
        minimising = solver.alg_params.minimising

    key_f = operator.attrgetter('fitness')
    if minimising:
        f = copy.deepcopy(sorted(pop, key=key_f, reverse=False))
    else:
        f = copy.deepcopy(sorted(pop, key=key_f, reverse=True))
    return f[:n]


# CROSSOVER
def crossover_one_point_saes(solver, par1, par2, xo_chance=None, point=None):
    """
    One point crossover for the SAES

    :param solver:          Solver used to start the optimisation
    :param par1:            Parent 1
    :param par2:            Parent 2
    :param xo_chance:       Crossover chance
    :param point:           Specific crossover point
    :return:                Trace object
    """
    assert len(par1.es_params) == len(par1.trace)
    assert len(par2.es_params) == len(par2.trace)
    if xo_chance is None:
        xo_chance = solver.alg_params.crossover_rate

    r = random.random()
    if r < xo_chance:
        if point is None:
            point = random.randrange(0, min(len(par1.trace), len(par2.trace))+1)
        new_trace = par1.trace[:point] + par2.trace[point:]
        new_es_params = par1.es_params[:point] + par2.es_params[point:]
        t = solver.create_solution_with_raw_trace(new_trace)
        new_es_params = fix_saes_params(solver, t.trace, new_es_params)
        t.es_params = new_es_params
    else:
        t = copy.deepcopy(random.choice([par1, par2]))
    assert len(t.es_params) == len(t.trace)
    return t


def fix_saes_params(solver, t, sds, default_sd=None):
    """
    Repair the strategy parameters (specifically for SAES)

    :param solver:          Solver used to start the optimisation
    :param t:               Trace containing solution
    :param sds:             Strategy parameters (standard deviations)
    :param default_sd:      Default mutation strength
    :return:
    """
    if default_sd is None:
        default_sd = solver.alg_params.mutation_sd
    # t = trace.trace
    sds = sds

    if len(t) < len(sds):
        sds = sds[:len(t)]
    elif len(t) > len(sds):
        for _ in t[len(sds):]:
            sds.append(default_sd)
    assert len(t) == len(sds)
    return sds


def crossover_one_point_saea(solver, par1, par2, xo_chance=None, point=None):
    if point is None:
        point = random.randrange(0, min(len(par1.trace), len(par2.trace)) + 1)
    t = crossover_one_point(solver, par1, par2, xo_chance, point)
    t.es_params = par1.es_params if point < 0.5 else par2.es_params
    return t


def crossover_one_point(solver, par1, par2, xo_chance=None, point=None):
    """
    One-point-crossover solution recombination

    :param solver:          Solver used to start the optimisation
    :param par1:            Parent 1
    :param par2:            Parent 2
    :param xo_chance:       Crossover chance
    :param point:           Specific crossover point
    :return:                Trace object
    """
    assert len(par1.trace) > 0
    assert len(par2.trace) > 0
    if xo_chance is None:
        xo_chance = solver.alg_params.crossover_rate
    r = random.random()
    if r < xo_chance:
        p1 = par1.trace
        p2 = par2.trace
        # print('p1')
        # print_raw_trace(p1)
        # print('p2')
        # print_raw_trace(p2)
        # print('')
        # Just in case traces are different lengths: (+1 to be inclusive for both edges)
        if point is None:
            point = random.randrange(0, min(len(p1), len(p2))+1)
        # print('cut at', point)
        # print(rounded_trace(p1[:point]), '  +  ', rounded_trace(p2[point:]), '\n')
        new_trace = p1[:point] + p2[point:]
        # print_raw_trace(new_trace)
        # print('')
        # print('Final sol:', rounded_trace(new_trace[sa_len:]))
        # print('Final sa:', rounded_trace(new_trace[:sa_len]))
        # return solver.create_solution_with_raw_trace(new_trace[sa_len:], new_trace[:sa_len])
        return solver.create_solution_with_raw_trace(new_trace)
    else:
        return copy.deepcopy(random.choice([par1, par2]))


# MUTATION
def mutate_trace(solver, t, mutation_chance=None):
    """
    Mutates a single value in the trace to a random value

    :param solver:          Solver used to start the optimisation
    :param t:               Trace object
    :param mutation_chance: Mutation chance
    :return:                Trace object
    """
    if mutation_chance is None:
        mutation_chance = solver.alg_params.mutation_rate

    r = random.random()
    if r < mutation_chance:
        c_trace = t.trace
        t_ind = random.randrange(len(c_trace))
        c_trace[t_ind] = random.random()
        t = solver.create_solution_with_raw_trace(c_trace)
    return t


def mutate_trace_gauss_saes(solver, t, sds, mutation_chance=None):
    """
    Mutates trace elements by an amount determined by a gaussian distribution.

    :param sds:                 Standard deviations for each individual gauss mutation
    :param solver:              Solver used to start the optimisation
    :param t:                   Trace object
    :param mutation_chance:     Chance of an element's mutation
    :return:                    Trace object
    """
    if mutation_chance is None:
        mutation_chance = solver.alg_params.mutation_rate

    p = t.trace
    for i in range(len(p)):
        r = random.random()
        if r < mutation_chance:
            # p[i] = sample_gauss(p[i], sds[i])
            # p[i] = p[i] + sds[i] * random.gauss(0, 1)
            # p[i] = max(0, min(p[i], 1))
            p[i] = get_trunc_norm(p[i], sds[i])
    t2 = solver.create_solution_with_raw_trace(p)
    t2.es_params = fix_saes_params(solver, t2.trace, t.es_params)
    return t2


def mutate_value(solver, val, mutation_chance=None, sd=0.1):
    """
    Mutates a single value. Sample from truncated Gaussian distribution

    :param solver:              Solver used to start the optimisation
    :param val:                 Number
    :param mutation_chance:     Chance of mutation
    :param sd:                  Mutation standard deviation
    :return:                    Number
    """
    if mutation_chance is None:
        mutation_chance = solver.alg_params.mutation_rate
    r = random.random()
    if r < mutation_chance:
        # val += sd * random.gauss(0, 1)
        val = get_trunc_norm(val, sd)
        # val = max(0.000001, min(val, 1.01))
    return val


def mutate_trace_gauss(solver, t, mutation_chance=None, sd=None):
    """
    Mutates trace elements by an amount determined by a gaussian distribution.

    :param solver:              Solver used to start the optimisation
    :param t:                   Trace object
    :param mutation_chance:     Chance of an element's mutation
    :param sd:                  Standard deviation for the gaussian distribution
    :return:                    Mutated Trace object
    """
    if mutation_chance is None:
        mutation_chance = solver.alg_params.mutation_rate
    if sd is None:
        sd = solver.alg_params.mutation_sd

    p = t.trace
    for i in range(len(p)):
        r = random.random()
        if r < mutation_chance:
            p[i] = get_trunc_norm(p[i], sd)
            # p[i] = p[i] + sd * random.gauss(0, 1)
            # p[i] = sample_gauss(p[i], sd)
            # p[i] = max(0, min(p[i], 1))
            # p[i] = sample_gauss(p[i], sd)
    t2 = solver.create_solution_with_raw_trace(p)
    t2.es_params = t.es_params
    return t2


def mutate_strategy(sds, lp=2.0):
    """
    Mutate each component of strategy parameters independently.
    Finally scale whole vector.

    :param sds:                 Strategy parameters
    :param lp:                  Learning parameter
    :return:                    Strategy parameters
    """
    t = math.sqrt(lp * len(sds)) ** -1.0
    tau = math.sqrt(lp * math.sqrt(len(sds))) ** -1.0
    child_sds = []
    for sd in range(len(sds)):
        n_sd = min(10, sds[sd] * (math.exp(tau*random.gauss(0, 1) + t*random.gauss(0, 1))))
        child_sds.append(n_sd)
    return child_sds


def sample_gauss(mean, sd, lower=0, upper=1):
    """
    Samples a gaussian distribution between 0 and 1.
    Gives a better approximation of a guassian distribution than clipping,
    but is less efficient. For a more accurate calculation, this could be
    replaced by scipy.stats.truncnorm

    :param mean:        mean
    :param sd:          standard deviation
    :return:            a float in range 0 to 1
    """
    r = random.gauss(mean, sd)
    while not (lower < r < upper):
        r = random.gauss(mean, sd)
    return r


def add_gauss_noise(val, sd, lower, upper):
    r = random.gauss(0, sd)
    while not (lower < r+val < upper):
        r = random.gauss(0, sd)
    return r+val


def get_sa_trace(t):
    return t.es_params + t.trace, len(t.es_params)


def get_sa_trace_for_2(t1, t2):
    return t1.es_params + t1.trace, t2.sa_params + t2.trace, len(t1.es_params)


def fitter_than(solver, t1, t2, minimising=None):
    if minimising is None:
        minimising = solver.alg_params.minimising
    if minimising:
        return t1.fitness <= t2.fitness
    else:
        return t1.fitness >= t2.fitness


def calc_average(pop):
    s = 0
    for p in pop:
        s += p.fitness
    return s / len(pop)


def print_raw_trace(t):
    rounded_list = [round(elem, 3) for elem in t]
    print('t  :', rounded_list)


def rounded_trace(t):
    return [round(elem, 3) for elem in t]


def print_trace(t):
    print_raw_trace(t.trace)
    if len(t.sa_params) > 0:
        print('sap:', t.sa_params)
    print('sol:', t.solution)
    print('fit:', t.fitness)
    print('')
