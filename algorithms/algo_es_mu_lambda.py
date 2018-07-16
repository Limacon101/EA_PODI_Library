from core import ops
import math
import copy


def run(solver, alg_params):
    """
    Simple ES. Base to build SAES
    """
    # Input: num_gens, pop_size, num_children
    verbose = alg_params.verbose

    pop_size = alg_params.pop_size
    num_gens = alg_params.generations
    num_children = 20  # lambda

    # Create Parent Population
    pop = []
    for p in range(pop_size):
        t = solver.create_solution()
        pop.append(t)
    best = ops.select_best_n(solver, pop, 1)[0]

    for g in range(num_gens):
        if verbose == 0:
            print('\nGeneration', g)
        child_pop = []
        for _ in pop:
            par1 = alg_params.select(solver, pop)
            par2 = alg_params.select(solver, pop)

            # Crossover to form child:
            child = ops.crossover_one_point(solver, par1, par2)
            child = copy.deepcopy(child)
            child = ops.mutate_trace_gauss(solver, child)
            child_pop.append(child)

            if verbose == 0:
                if solver.generator.__name__ == 'gen_gp':
                    print(' s:', g, solver.generator.get_formula(child.solution), '')
                else:
                    print(' s:', g, child.solution, '  f:', child.fitness)

        new_pop = [ops.select_tournament(solver, pop + child_pop) for _ in range(num_children)]
        pop = new_pop

        c_best = ops.select_best_n(solver, new_pop, 1)[0]
        best = ops.select_best_n(solver, [best, c_best], 1)[0]
        solver.stats.update_stats(best, c_best, ops.calc_average(new_pop))

        if verbose >= 0:
            if verbose == 0 or g % verbose == 0 or g == num_gens-1:
                print('\nGeneration', g, 'av: ', calc_average(new_pop))
                print('Best solution:   ', c_best.fitness)
                print('                 ', c_best.trace)

    return best


def calc_average(pop):
    s = 0
    for p in pop:
        s += p.fitness
    return s / len(pop)
