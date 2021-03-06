from core import ops
import math

elitist = True


def run(solver, alg_params):
    pop_size = alg_params.pop_size
    num_children = alg_params.num_children
    num_gens = alg_params.generations
    verbose = alg_params.verbose

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
        for par in range(pop_size):
            # Select parents:
            par1 = alg_params.select(solver, pop)
            par2 = alg_params.select(solver, pop)

            # Crossover to form child:
            child = ops.crossover_one_point(solver, par1, par2)

            # Chance to mutate child:
            child = alg_params.mutate(solver, child)
            child_pop.append(child)
            if verbose == 0:
                if solver.generator.__name__ == 'gen_gp':
                    print(' s:', par, solver.generator.get_formula(child.solution), '')
                else:
                    print(' s:', par, child.solution, '  f:', child.fitness)

        # Sort children and keep top half:
        if elitist:
            best_children = ops.select_best_n(solver, child_pop, num_children)
        else:
            best_children = [ops.select_tournament(solver, child_pop) for _ in range(num_children)]
        c_best = ops.select_best_n(solver, best_children, 1)[0]

        if verbose >= 0:
            if verbose == 0 or g % verbose == 0 or g == num_gens-1:
                print('\nGeneration', g, 'av: ', ops.calc_average(best_children))
                print('Best solution:   ', c_best.fitness)
                print('                 ', c_best.trace)
        best = ops.select_best_n(solver, [best, c_best], 1)[0]
        solver.stats.update_stats(best, c_best, ops.calc_average(best_children))
        pop = best_children
    return best


def print_pop(m_pop):
    for t in m_pop:
        # print("sol:", t.solution, " trace:", t.trace, "  f:", f)
        print("sol:", t.solution, "  f:", t.fitness)
    print('')


def print_t(t):
    # print("s/t:", t.solution, t.trace, "  f:", t.fitness)
    print("s:", t.solution, "  f:", t.fitness)

