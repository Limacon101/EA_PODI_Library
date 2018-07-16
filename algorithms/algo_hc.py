import copy
from core import ops

more_gens = False


def run(solver, alg_params):
    """
    Simple Hill Climber
    """
    # more_gens -- number of generations = gens*pop_size to make for fair comparison with EA
    gens = alg_params.generations if not more_gens else alg_params.generations * int(alg_params.pop_size)
    verbose = alg_params.verbose

    t = solver.create_solution()
    best_t = copy.deepcopy(t)

    for g in range(gens):
        # t = operators.mutate_trace(solver, t)
        t = alg_params.mutate(solver, t)
        if ops.fitter_than(solver, t, best_t):
            best_t = copy.deepcopy(t)
        else:
            t = copy.deepcopy(best_t)

        if (more_gens and g % alg_params.pop_size == 0) or not more_gens:
            solver.stats.update_stats(best_t)

        if verbose >= 0:
            if verbose == 0 or g % verbose == 0 or g == gens-1:
                print('\nIteration', g, 'best: ', best_t.fitness)

    return best_t
