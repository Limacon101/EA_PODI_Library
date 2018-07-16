from algorithms import algo_saea
from core import ops, graphing
import sys
from generators import gen_onemax, gen_schwefel, gen_gp, gen_sphere, gen_rastrigin
from core import solver
import random

from matplotlib import pylab as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np

best_es_params = []


def run(solver, alg_params):
    """
    Self-Adaptive Evolutionary Strategy. Mutates solution variable's mutation strength
    """
    global best_es_params

    verbose = alg_params.verbose
    pop_size = alg_params.pop_size
    num_gens = alg_params.generations
    num_children = alg_params.num_children  # lambda

    # Create Parent Population
    pop = []
    for p in range(pop_size):
        t = solver.create_solution()
        # strategy = create_random_strategy(len(t.trace))
        strategy = create_set_strategy(len(t.trace), 0.1)
        t.es_params = strategy
        pop.append(t)
    # best = ops.select_best_n(solver, pop, 1)[0]
    best = pop[0]

    # Evolutionary process
    for g in range(num_gens):
        child_pop = []
        for c in range(num_children):
            par1 = alg_params.select(solver, pop)
            par2 = alg_params.select(solver, pop)

            # Crossover to form child:
            child = ops.crossover_one_point_saes(solver, par1, par2)

            # Mutate strategy parameters:
            child.es_params = ops.mutate_strategy(child.es_params)

            # Mutate objective parameters:
            child = ops.mutate_trace_gauss_saes(solver, child, child.es_params)
            child_pop.append(child)

        # Note: Choose plus or comma ES here:
        # new_pop = [ops.select_tournament(solver, pop + child_pop) for _ in range(num_children)]
        new_pop = [ops.select_tournament(solver, child_pop) for _ in range(num_children)]
        # new_pop = ops.select_best_n(solver, child_pop, num_children)

        c_best = ops.select_best_n(solver, new_pop, 1)[0]
        best = ops.select_best_n(solver, [best, c_best], 1)[0]
        solver.stats.update_stats(best, c_best, ops.calc_average(new_pop))
        # print(best.fitness, end=' ')
        best_es_params.append(best.es_params)

        pop = new_pop
        # print('\nPop:', g)
        # for p in pop:
        #     print(p.es_params)
    # global best_es_params = best.es_params
    # print('Best es_params:', best.es_params)
    return best


def create_random_strategy(n):
    return [random.random() * 0.05 for _ in range(n)]


def create_set_strategy(n, f):
    return [f for _ in range(n)]


def graph_es_params():
    alg = sys.modules[__name__]
    alg = algo_saea
    gen = gen_schwefel
    # gen.n = 5
    alg_params = solver.AlgoParams(minimising=True, generations=500,
                                   mutation_rate=0.15,
                                   mutation_sd=0.15)
    print('alg_params from saes:')
    print(alg_params)

    my_s = solver.Solver(gen, alg, alg_params)
    s = my_s.solve()
    print(s.fitness)

    best_es_params = alg.best_es_params
    # Best es params contains list of params at each generation
    print('best es params:')
    print(best_es_params)
    print()

    fig = plt.figure()
    xs = list(range(0, alg_params.generations))
    print('xs:', xs)

    av_params = []
    for d in best_es_params:
        av_params.append(sum(d) / len(d))

    plt.grid(True)
    plt.plot(xs, av_params)
    plt.title('SAES\'s Self-Adaption of its Mutation Standard Deviation\n'
              'Strategy Parameters for the Sphere Problem over 500 Generations')
    plt.title('SAEA\'s Self-Adaption of its Mutation Rate Parameter\n'
              'for Schwefel\'s Problem over 500 Generations')
    plt.xlabel('Generations')
    plt.ylabel('Average Strategy Parameter Value')
    plt.ylabel('Mutation Rate Parameter Value')
    plt.yscale('log')

    plt.xlim(0, alg_params.generations)

    plt.show()


if __name__ == '__main__':
    # Generate Histogram of solution fitness ranges:

    alg = sys.modules[__name__]
    gen = gen_schwefel
    aps = solver.AlgoParams(generations=500,
                            pop_size=30,
                            num_children=25,
                            crossover_rate=0.2,
                            mutation_rate=0.15)
    n_runs = 50
    # res = []
    # for n in range(n_runs):
    #     s = solver.Solver(gen, alg, aps).solve()
    #     print(n, ':', s.fitness)
    #     res.append(s.fitness)
    # print(res)
    # res = [2.2737367544323206e-12, 118.43838755855404, 118.43856955173192, 118.43835520631183, 113.57959969417311, 118.43833472930169, 2.864908310584724e-11, 0.00016502215703440015, 355.31500391488316, 5.066203812020831e-08, 1.0488085990800755e-05, 118.4383346144391, 1.426028484274866e-05, 3.1582203519064933e-09, 236.87666922887684, 236.87666922888866, 0.0008786196622168063, 1.1000338417943567e-09, 118.43833905039901, 9.959444469131995e-07, 9.094947017729282e-13, 2.256047082482837e-08, 118.43886998615108, 118.43833461443865, 1.3642420526593924e-12, 118.4383399793885, 118.43833461652503, 236.87666925612712, 0.002691033574137691, 1.3642420526593924e-12, 118.44446094780551, 2.0776451492565684e-07, 1.9784692995017394e-08, 118.4383346255338, 434.2793387811553, 0.0005252962891972857, 296.1067141487724, 8.712272574484814e-07, 1.4633769751526415e-09, 217.13968023843586, 2.6299443106836407e-05, 335.5780040050322, 296.10673607587887, 4.1382008930668235e-10, 5.9117155615240335e-12, 1.4443594409385696e-06, 118.4383346144757, 118.43833474164671, 3.6834535421803594e-11, 0.6714809380332554]
    # SAES
    res = [217.13966939134025, 0.0001614493044144183, 2.2458002604253124e-06, 118.43833596765626, 118.43842625118327, 5.263206276140409e-06, 8.827947294776095e-05, 118.43833461444342, 7.359085429925472e-07, 3.3108335628639907e-07, 5.536398930416908e-07, 118.4577605038155, 217.1396694134296, 2.5832767196989153, 118.4383346224422, 236.87667070148268, 118.4383878160636, 0.00047248668624888523, 4.4310581870377064e-08, 0.014323641458759084, 3.175282154188608e-05, 0.010152224094781559, 118.43836565701986, 118.43835997094766, 217.13967049932398, 335.5780075576447, 1.6131734810187481e-06, 118.43833461443842, 118.43833461862346, 2.7998665700579295e-05, 9.094947017729282e-13, 0.0048248189905280015, 118.4476237146173, 217.139671988778, 0.005378006960654602, 1.244688974111341e-08, 0.006369620808527543, 1.496118784416467e-10, 118.43833634515568, 1.3642420526593924e-12, 1.8289892977918498e-07, 7.017657480901107e-06, 118.43834876337542, 2.5784174795262516e-10, 118.43833515837241, 8.845623824527138e-05, 2.321030478924513e-09, 4.2599822336342186e-08, 5.8877958508674055e-08, 118.43833462009934]
    npres = np.asarray(res)
    res_med = np.median(npres, axis=0)
    print(max(res))

    sns.distplot(res, kde=False, rug=True, bins=10 ** np.linspace(np.log10(1e-12), np.log10(1e3), 31), label='saes')
    plt.axvline(res_med, color='blue', linestyle='dashed', linewidth=1)

    # EA
    res2 = [0.030177499048932077, 0.03182897475289792, 0.9067675879427952, 0.1255823789861097, 0.7304041154470724, 0.24013429351316518, 0.047395624610089726, 0.33455976182540326, 0.5122912110423385, 0.04053020926812678, 0.5490208242440531, 0.03095340965091964, 0.3634256622481189, 0.1735488362792239, 0.06583433567493557, 0.4196168630728607, 0.0791040294452614, 0.29585074026863367, 0.1570712770017053, 0.300995351716665, 0.516826878430038, 0.06373598526397473, 0.7570401502339337, 0.10004739449095723, 0.096699957866349, 0.18150070534557017, 0.13396922108995568, 0.021363969266985805, 0.16764719247476023, 0.18638590874115835, 0.6116211015751105, 0.23656649040685807, 0.2007282131521606, 0.5734442263392339, 0.42697947955957716, 0.126378931252475, 0.14140844533721975, 0.1481725490439203, 0.6364223072341701, 0.19464976195558847, 0.7164994727909288, 0.11666023308043805, 0.13603975788464595, 0.36837669094165904, 2.135818799550634, 0.204471031378489, 0.022981775069638388, 0.21821436884556533, 0.24699250618505175, 0.5899182523762647]
    npres2 = np.asarray(res2)
    res2_med = np.median(npres2, axis=0)

    sns.distplot(res2, kde=False, rug=True, bins=10 ** np.linspace(np.log10(1e-12), np.log10(1e3), 31), label='ea')
    plt.axvline(res2_med, color='red', linestyle='dashed', linewidth=1)

    plt.legend(loc='upper left')

    # plt.hist(res, bins=10 ** np.linspace(np.log10(1e-12), np.log10(1e3), 30))
    plt.gca().set_xscale("log")

    # n, bins, patches = plt.hist(res, 10, normed=1, alpha=0.8)
    plt.title('Distribution of Solution Fitness\' for Schwefel\'s problem')
    plt.xlabel('Final Solution Fitness')
    plt.ylabel('Frequency')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.show()

    # for d in best_es_params:
    #     plt.plot(d, color="blue", linewidth=1.5, linestyle="-", label='Overall Best')
    #

    # for g in range(alg_params.generations):
    #     for di, d in enumerate(best_es_params[g]):
    #         grad = di / len(best_es_params[g])
    #         # plt.scatter(g, d, marker='+', color=plt.cm.RdYlBu(grad))
    #         sns.jointplot(x=g, y=d, kind='hex')

    # sns.jointplot(x=alg_params.generations, y=best_es_params[0], kind='hex')

    # x = []
    # y = []
    # for d in best_es_params:
    #
    #
    # x = alg_params.generations
    # y = best_es_params[0]
    # l1 = [l[0] for l in best_es_params]
    #
    # x = np.asarray(list(range(0, alg_params.generations)))
    # y = np.asarray(l1)
    # with sns.axes_style("white"):
    #     sns.jointplot(x=x, y=y, kind="hex", color="k")
    #
    # # plt.hexbin(np.arange(0, 0.02, 1), y)
    #
    # print('xtype:', type(x))
    # plt.show()

    # res = []
    # for z in range(1):
    #     my_s = solver.Solver(gen, alg, alg_params)
    #     s = my_s.solve()
    #     # print('Best', my_s.stats.best)
    #     # print()
    #     # print('gen av:', my_s.stats.gen_average)
    #     # print()
    #     # print('gen best:', my_s.stats.gen_best)
    #     # print()
    #     # print('num gens:', my_s.stats.num_gens)
    #     print(s.fitness)
    #     print('es_params:', best_es_params)
    #     res.append(s.fitness)

    # graphing.graph_fitness_vs_generations(my_s.stats.gen_average)
    # graphing.graph_genav_genbest_best_vs_generations(my_s.stats.gen_average,
    #                                                  my_s.stats.gen_best,
    #                                                  my_s.stats.best)
    # res.sort()
    # print(res)
    # for r in range(len(res)):
    #     continue
    #     print(r, ':', res[r])

    # sds = [0.9, 0.9, 0.9]
    # n_sds = mutate_strategy(sds)
    # print(n_sds)

    # Test fix sds method:
    # trace = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # sds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    #
    # n_sds = ops.fix_saes_params(trace.trace, sds, 0.1)
    # print('sds:', n_sds)
    # print('t:  ', trace)

    # Average each algorithms's results
    # Do all tests, then show graphs
    # Choose parameter tuning
    # Log scales
