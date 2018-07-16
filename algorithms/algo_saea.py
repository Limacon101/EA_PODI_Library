import sys
from generators import gen_onemax, gen_schwefel
from core import solver, ops
import random

def run(solver, alg_params):
    """
    Self-adaptive EA. Encodes mutation rate within each solution.
    """

    pop_size = alg_params.pop_size
    num_children = alg_params.num_children
    num_gens = alg_params.generations
    mut_chance = alg_params.mutation_rate
    mut_sd = alg_params.mutation_sd
    verbose = alg_params.verbose

    # Create Parent Population
    pop = []
    for p in range(pop_size):
        t = solver.create_solution()
        t.es_params = [mut_chance]
        pop.append(t)
    best = ops.select_best_n(solver, pop, 1)[0]

    for g in range(num_gens):
        if verbose == 0:
            print('\nGeneration', g)

        child_pop = []
        for par in range(pop_size):
            # Select parents:
            par1 = alg_params.select(solver, pop)
            par2 = alg_params.select(solver, pop)

            # Crossover to form child:
            child = ops.crossover_one_point_saea(solver, par1, par2)

            # Chance to mutate child:
            child.es_params = [ops.mutate_value(solver, child.es_params[0], mutation_chance=child.es_params[0])]
            child = alg_params.mutate(solver, child, mutation_chance=child.es_params[0])
            # child.es_params = [ops.mutate_value(solver, child.es_params[0], sd=child.es_params[0])]
            # child = alg_params.mutate(solver, child, sd=child.es_params[0])

            child_pop.append(child)
            if verbose == 0:
                if solver.generator.__name__ == 'gen_gp':
                    print(' s:', par, solver.generator.get_formula(child.solution), '')
                else:
                    print(' s:', par, child.solution, '  f:', child.fitness)

        # Sort children and keep top half:
        best_children = [ops.select_tournament(solver, child_pop) for _ in range(num_children)]
        c_best = ops.select_best_n(solver, best_children, 1)[0]

        if verbose >= 0:
            if verbose == 0 or g % verbose == 0 or g == num_gens - 1:
                print('\nGeneration', g, 'av: ', calc_average(best_children))
                print('Best solution:   ', c_best.fitness)
                print('                 ', c_best.trace)
        best = ops.select_best_n(solver, [best, c_best], 1)[0]
        # print('m:', best.es_params[0])

        solver.stats.update_stats(best, c_best, ops.calc_average(best_children))
        pop = best_children
    # print('')
    # print('m:', best.es_params[0])
    return best


def calc_average(pop):
    s = 0
    for p in pop:
        s += p.fitness
    return s / len(pop)


def print_pop(m_pop):
    for t in m_pop:
        # print("sol:", t.solution, " trace:", t.trace, "  f:", f)
        print("sol:", t.solution, "  f:", t.fitness)
    print('')


def print_t(t):
    # print("s/t:", t.solution, t.trace, "  f:", t.fitness)
    print("s:", t.solution, "  f:", t.fitness)

