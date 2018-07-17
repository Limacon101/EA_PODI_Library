"""
Module provides examples of more extensive library usage (see the __main__ method for usage).

This includes:
 - Running a simple optimisation
 - Averaging multiple optimisation results
 - Graphing multiple Algorithm's performance on chosen problems
 - Graphing heat-maps to show algorithm performance over a range of parameter combinations
"""

import numpy as np
from algorithms import algo_ea, algo_hc, algo_rs, algo_es_mu_lambda, algo_es_mu_lambda, algo_saes
from algorithms import algo_saea
from generators import gen_gp, gen_onemax, gen_rastrigin, gen_schwefel, gen_sphere
from core import solver, ops, graphing


def graph_multiple_algos_performance(algos, gen, num_runs):
    """
    Plot performance comparison graphs for a range of algorithms optimising a
    particular problem generator. Prints out results.
    Specific parameters are obtained from systematic tuning process of mutation
    rate and mutation sd that was undertaken using the t_parameter_tuning method.

    :param algos:           List of algorithms
    :param gen:             Generator
    :param num_runs:        Number of runs
    :return:
    """
    aps = solver.AlgoParams(generations=500,
                            pop_size=30,
                            num_children=25,
                            crossover_rate=0.2,
                            mutation_rate=0.15)
    title = 'Algorithm Progress Comparison for '
    log = True
    legend_loc = 'lower left'
    algo_hc.more_gens = False
    algo_ea.elitist = False
    av_type = 'mean'
    specific_params = None

    # Specific parameter setup:
    if gen == gen_schwefel:
        specific_params = [{"mutation_rate": 0.10, "mutation_sd": 0.35},
                           {"mutation_rate": 0.15, "mutation_sd": 1.00},
                           {"mutation_rate": 1.00, "mutation_sd": 0.50},
                           {"mutation_rate": 0.75, "mutation_sd": 0.25},
                           {"mutation_rate": 1.00, "mutation_sd": 1.00}]
        aps.mutation_rate = 0.1
        aps.mutation_sd = 0.35
        aps.minimising = True
        title += 'Schwefel\'s Problem'
        log = True
        legend_loc = 'lower left'
        av_type = 'median'
    if gen == gen_rastrigin:
        specific_params = [{"mutation_rate": 0.05, "mutation_sd": 0.15},
                           {"mutation_rate": 0.30, "mutation_sd": 1.00},
                           {"mutation_rate": 1.00, "mutation_sd": 0.50},
                           {"mutation_rate": 0.10, "mutation_sd": 0.10},
                           {"mutation_rate": 1.00, "mutation_sd": 1.00}]
        aps.mutation_rate = 0.05
        aps.mutation_sd = 0.15
        aps.minimising = True
        title += 'Rastrigin\'s Problem'
        log = True
        legend_loc = 'lower left'
        av_type = 'mean_with_offset'
    if gen == gen_gp:
        specific_params = [{"mutation_rate": 0.15, "mutation_sd": 0.30},
                           {"mutation_rate": 0.30, "mutation_sd": 1.00},
                           {"mutation_rate": 1.00, "mutation_sd": 0.15},
                           {"mutation_rate": 0.10, "mutation_sd": 0.10},
                           {"mutation_rate": 1.00, "mutation_sd": 1.00}]
        aps.mutation_rate = 0.15
        aps.mutation_sd = 0.3
        aps.minimising = True
        title += 'the GP Problem'
        log = True
        legend_loc = 'upper right'
        av_type = 'mean'
    if gen == gen_sphere:
        specific_params = [{"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 1.00, "mutation_sd": 1.00}]
        aps.mutation_rate = 0.05
        aps.mutation_sd = 0.05
        aps.minimising = False
        title += 'the Sphere Problem'
        log = False
        legend_loc = 'lower right'
        algo_hc.more_gens = True
        algo_ea.elitist = False
        av_type = 'mean_with_offset'
    if gen == gen_onemax:
        specific_params = [{"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 0.05, "mutation_sd": 1.00},
                           {"mutation_rate": 1.00, "mutation_sd": 0.05},
                           {"mutation_rate": 0.05, "mutation_sd": 0.05},
                           {"mutation_rate": 1.00, "mutation_sd": 1.00}]
        aps.mutation_rate = 0.05
        aps.mutation_sd = 0.05
        aps.minimising = False
        title += 'the OneMax Problem'
        log = False
        legend_loc = 'lower right'
        algo_hc.more_gens = True
        algo_ea.elitist = False
        av_type = 'mean_with_offset'

    print(title)
    print()
    print(aps)
    print('log:', log)
    print('elitest:', algo_ea.elitist)

    run_with_stats2(gen, algos, aps, num_runs, log=log, ind_params=specific_params, av_type=av_type,
                    title=title, legend_loc=legend_loc)


def run_n_times(gen, alg, aps, n, print_ind=False, print_alg=False):
    """
    Average the results from multiple optimisation runs of the same
    algorithm/problem generator combination

    :param gen:                 Problem generator
    :param alg:                 Optimisation algorithm
    :param aps:                 Algorithm parameters
    :param n:                   Number of optimisations to average over
    :param print_ind:           Print each individual result
    :param print_alg:           Print Algorithm name
    :return:                    Average optimisation fitness result
    """
    if print_alg:
        print('\n', alg.__name__)
    res = []
    for i in range(n):
        s = solver.Solver(gen, alg, aps).solve()
        if print_ind:
            print(i, ':', s.fitness)
            # print(i, ':', s.fitness, '     s:', gen_gp.get_formula(s))

        res.append(s.fitness)
    av = sum(res) / float(len(res))
    print(' Av:', av, '        sd:', aps.mutation_sd, 'r:', aps.mutation_rate)
    return av


def compare_algs(gen, algos, aps, n, print_ind=False):
    """ Calls run_n_times for a number of different algorithms """
    for a in algos:
        run_n_times(gen, a, aps, n, print_ind)


def run_with_stats(gen, algos, aps, n_runs, log=True, title='Alg. Performance comparison', path='alg_progress/',
                   legend_loc='upper left'):
    """
    Outputs a graph showing the generational best of each algorithm's solutions
    over a number of generations. This involves accessing the statistics of the
    algorithm throughout the optimisation process

    :param gen:                 Problem generator
    :param algos:               List of algorithms
    :param aps:                 Algorithm parameters
    :param n_runs:              Number of runs
    :param log:                 Use a log scale for the (graphed) fitness value
    :param title:               Graph title
    :param path:                Path to output graph image to
    :param legend_loc:          Legend location
    :return:
    """
    print('\n', gen.__name__)
    bests_dict = {}
    for alg in algos:
        print('\n', alg.__name__)
        n = 0
        means = None
        for run in range(n_runs):
            n += 1
            s = solver.Solver(gen, alg, aps)
            sol = s.solve()
            print(run, ':', sol.fitness)
            # Get generational statistics:
            n_means = s.stats.get_stats()
            n_means = tuple(i for i in n_means if len(i) > 0)
            if means is None:
                means = [[0] * aps.generations] * len(n_means)
            # Add statistics to an iterative mean:
            for b in range(len(n_means)):
                for stat_type in range(len(n_means)):
                    means[stat_type][b] += (n_means[stat_type][b] - means[stat_type][b]) / n_runs
        mean_bests = n_means[0]
        bests_dict[alg.__name__] = mean_bests
        print('Av Best:', mean_bests[-1])
        print('Mean bests:')
        print(mean_bests)

        # mean_gbests = n_means[1] if len(n_means) > 1 else None
        # mean_gavs = n_means[2] if len(n_means) > 2 else None
        # -- Graph bests, genav, genbest for each alg --
        # fig_title = 'Problem-' + gen.__name__[15:] + ', Algorithm-' + alg.__name__[16:] + ' (' + str(
        #     aps.generations) + ' generations)'
        # graphing.graph_genav_genbest_best_vs_generations(mean_bests, mean_gbests, mean_gavs, title=fig_title,
        #                                                  path=path)

        # print('\nmean_bests  :', mean_bests)
        # print('mean_gen_bests:', mean_gbests)
        # print('mean_gen_avs  :', mean_gavs)
        # print()
        # for b in range(len(bests)):
        #     mean_bests[run][b] += (bests[b] - mean_bests[b]) / n_runs
    # title = "Algorithm Progress Comparison for the OneMax Problem"
    graphing.graph_genbest_vs_generations_many_algs(bests_dict, log=log, title=title, legend_loc=legend_loc)


def run_with_stats2(gen, algos, aps, n_runs, log=True, ind_params=None, av_type='mean_with_offset',
                    title='Alg. Performance comparison', path='alg_progress/', legend_loc='upper left'):
    """
    Similar to run_with_stats(), except different average types can be used
    (mean_with_offset, mean, median).
    Mean with offset particularly useful, as anomalies can skew performance results
    if just the mean is used
    """
    # Only graphs mean bests:
    print('\n', gen.__name__)
    bests_dict = {}

    # Remove anomalies (from both best and worst)
    c_offset = 2  # Offset from centre
    offset = (n_runs // 2) - c_offset
    if (n_runs - offset*2) <= 1:
        print('Offset too large at:', offset, 'for n_runs:', n_runs)
        return

    for a, alg in enumerate(algos):
        print('\n', alg.__name__)
        all_means = np.zeros((n_runs, aps.generations))
        if ind_params is not None:
            aps.mutation_rate = ind_params[a]['mutation_rate']
            aps.mutation_sd = ind_params[a]['mutation_sd']
            print('Specific params:')
            print(ind_params[a])

        # Multiple runs:
        for run in range(n_runs):
            s = solver.Solver(gen, alg, aps)
            sol = s.solve()
            print(' ', run, ': fitness', sol.fitness)

            # Get generational statistics:
            bests = np.asarray(s.stats.get_generational_overall_bests())
            all_means[run] = bests
            # means = np.add(bests, means)
            # n_means = s.stats.get_stats()
            # n_means = tuple(i for i in n_means if len(i) > 0)
            # if means is None:
            #     means = [[0] * aps.generations] * len(n_means)
            # # Add statistics to an iterative mean:
            # for b in range(len(n_means)):
            #     for stat_type in range(len(n_means)):
            #         means[stat_type][b] += (n_means[stat_type][b] - means[stat_type][b]) / n_runs

        # Sort by last column (the final best score)
        all_means = all_means[all_means[:, -1].argsort()]

        print()
        print('Matrix of all trial generational progress: rows-trials, columns-generations')
        print(all_means)

        # Calculate Average
        if av_type == 'median':
            av_bests = all_means[n_runs // 2]
        elif av_type == 'mean':
            av_bests = np.mean(all_means, axis=0)
        elif av_type == 'mean_with_offset':
            all_means = all_means[offset:n_runs - offset]
            print('Matrix with offset:', offset, '(removed both best and worst)')
            print(all_means, '\n')
            av_bests = np.mean(all_means, axis=0)

        # med_bests = np.median(all_means, axis=0)
        # av_bests = means / n_runs
        # av_bests = med_bests
        print('Av bests:', av_bests.tolist())

        bests_dict[alg.__name__] = av_bests.tolist()
        print('Av Best:', av_bests[-1])
        # print('Med Best:', med_bests[-1])

    graphing.graph_genbest_vs_generations_many_algs(bests_dict, log=log, title=title, legend_loc=legend_loc)


def t_run_graphs():
    """ Call run_with_stats() [graphs performance comparisons] for a range of algorithms """
    alg_params = solver.AlgoParams(generations=500,
                                   pop_size=30,
                                   num_children=20,
                                   mutation_rate=0.1,
                                   crossover_rate=0.2)
    algs = [algo_hc, algo_saes, algo_ea, algo_es_mu_lambda]
    # algs = [algo_ea]
    gens = [gen_gp]
    run_with_stats(gens, algs, alg_params, 15, path='parameter_testing/')


def t_run_av_scores():
    """
    Call compare_algs() [prints average fitness results for range of algorithms] for
    multiple generators
    """
    alg_params = solver.AlgoParams(generations=500,
                                   pop_size=30,
                                   num_children=20,
                                   mutation_rate=0.1,
                                   crossover_rate=0.2)
    # algs = [algo_hc, algo_es_sa, algo_ea, algo_es_mu_lambda2]
    algs = [algo_ea]
    # gens = [gen_gp, gen_schwefel, gen_rastrigin, gen_sphere]
    gens = [gen_schwefel]
    for gen in gens:
        print()
        print()
        print(gen.__name__, ' D:', gen.n)
        compare_algs(gen, algs, alg_params, 15, True)


def t_parameter_tuning():
    """
    Plot heatmap showing the effects of different combinations of mutation rate and mutation
    strength on an algorithm's performance
    """
    num_runs = 25
    alg_params = solver.AlgoParams(generations=500,
                                   pop_size=20,
                                   num_children=15,
                                   mutation_rate=0.1,
                                   mutation_sd=0.1,
                                   crossover_rate=0.2,
                                   minimising=True)
    algs = [algo_hc]
    gens = [gen_schwefel]

    # Range of mutation rates/strengths -- focused on lower values
    m_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    m_sds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    for gen in gens:
        print('\n\n', gen.__name__, ' D:', gen.n, ' Av over:', num_runs, ' minimising:', alg_params.minimising)
        for alg in algs:
            print('\n', alg.__name__)
            m = build_parameter_matrix(m_sds, m_rates, gen, alg, alg_params, num_runs)
            # m = np.asarray([[3637.16, 3034.76, 4495.92, 3532.88, 3898.8, 3754.0, 3423.8, 3590.84, 3778.76, 3543.0, 3393.16, 3714.84], [3346.72, 1424.24, 1096.68, 1364.52, 924.8, 1097.0, 481.12, 919.24, 969.08, 678.84, 379.48, 184.12], [3104.6, 621.24, 517.2, 1147.6, 572.68, 504.0, 757.96, 205.72, 409.56, 514.48, 273.44, 389.6], [4462.68, 702.92, 907.28, 581.32, 682.72, 575.04, 260.8, 626.2, 406.36, 323.44, 173.28, 148.28], [3458.92, 610.0, 355.76, 425.24, 389.72, 291.6, 341.92, 281.96, 165.8, 444.68, 351.28, 193.56], [3619.6, 478.68, 266.04, 435.92, 357.44, 287.4, 217.68, 143.04, 209.48, 413.0, 277.8, 262.24], [3161.52, 364.4, 284.68, 142.56, 326.04, 280.32, 409.32, 97.2, 149.24, 222.2, 149.84, 223.2], [3662.76, 173.08, 270.64, 82.28, 235.44, 387.4, 138.32, 184.84, 381.8, 289.12, 59.36, 231.28], [3447.12, 119.08, 235.16, 98.0, 88.64, 187.92, 146.2, 155.2, 79.4, 374.68, 125.72, 123.6], [4175.0, 218.6, 133.52, 120.8, 107.88, 172.68, 223.56, 283.32, 173.76, 382.56, 56.44, 173.68], [4013.56, 175.48, 201.76, 204.56, 192.04, 185.6, 198.16, 158.16, 224.72, 114.36, 69.68, 148.52], [4163.6, 179.2, 168.64, 99.56, 85.24, 189.16, 128.76, 100.16, 191.72, 293.12, 177.6, 190.32]])
            # print(m.tolist())
            # fig_title = 'Parameter Tuning: Problem-' + gen.__name__[15:] + ', Algorithm-' + alg.__name__[
            #                                                                                 16:] + ' (' + str(
            #     alg_params.generations) + ' gens)'
            # # fig_title = 'Parameter Tuning. Problem:' + gen.__name__[15:] + ', Algorithm:' + alg.__name__[16:]
            # f_name = 'param_tuning ' + gen.__name__[15:] + ' ' + alg.__name__[16:] + ' gens ' + str(
            #     alg_params.generations)
            # graphing.graph_parameter_tuning_matrix(m_sds, m_rates, m, save=True, title=fig_title,
            #                                        xy_labels=('Mutation SD', 'Mutation Rate'), f_name=f_name)
            plot_params_from_existing_matrix(m, alg_params, m_rates, m_sds, gen, alg)


def plot_params_from_existing_matrix(m, alg_params, m_rates, m_sds, gen, alg):
    alg_params = solver.AlgoParams(generations=500,
                                   pop_size=20,
                                   num_children=15,
                                   mutation_rate=0.1,
                                   mutation_sd=0.1,
                                   crossover_rate=0.2,
                                   minimising=False)
    m = [[10.11017356, 9.81553943, 9.51936055, 9.82523074, 9.90350557, 10.41323724
             , 10.0431089, 10.19401907, 9.98654096, 10.2330111, 10.38928044, 9.88082642],
         [9.73675741, 29.92769755, 29.85296314, 29.66458991, 29.37557911, 29.02490396
             , 28.52751, 28.21897649, 27.87297921, 27.41085533, 25.74743397, 25.08453027],
         [9.82730768, 29.86759836, 29.73365104, 29.36092358, 28.87696313, 28.30851731
             , 27.66110661, 26.9878795, 26.35847077, 25.34069757, 23.06773356, 21.43972092],
         [10.26633115, 29.79890697, 29.61862317, 29.06128894, 28.21519508, 27.37523308
             , 26.45882201, 25.68030397, 25.00117499, 23.62470683, 21.09191409, 19.16926496],
         [10.63998782, 29.74417701, 29.46870148, 28.77817534, 27.76882605, 26.7717329
             , 25.6994397, 24.74414862, 23.72830812, 22.34229872, 19.42088004, 17.64807017],
         [10.36964505, 29.65099585, 29.33813597, 28.52330292, 27.43522808, 25.86128857
             , 24.67195874, 23.84316289, 23.17486243, 21.36591445, 18.44826842, 16.55066666],
         [9.80103791, 29.63947407, 29.22615162, 28.33837772, 27.09110581, 25.51645281
             , 24.39930918, 23.35728752, 22.35774169, 20.89962661, 18.13534053, 16.0687385, ],
         [9.7370132, 29.59317577, 29.13148613, 28.14169556, 26.61791957, 25.48137308
             , 24.11631185, 22.85133132, 22.01330386, 20.54354845, 17.58273639, 15.90476485],
         [9.74284256, 29.52116474, 29.07085953, 27.9268989, 26.54639009, 25.00716752
             , 23.9372031, 22.87836236, 21.65145519, 20.43217555, 17.67406338, 15.60329239],
         [9.39964721, 29.49094221, 28.98298751, 27.79543022, 26.35906484, 24.95810682
             , 23.64452962, 22.86325525, 21.77494932, 20.34533931, 17.66264358, 15.86921516],
         [9.75688406, 29.44039586, 28.97745966, 27.86386639, 26.54911583, 24.99166255
             , 24.07931692, 22.96002054, 22.16953415, 20.36124801, 18.03818508, 16.18856926],
         [10.17113579, 29.47464897, 29.00695665, 27.95192625, 26.45901391, 25.48240373
             , 23.8780808, 23.20892586, 22.4747484, 20.73948254, 18.37320129, 16.45568735]]
    m = np.asarray(m)
    print('m shape:', m.shape)
    m_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    m_sds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    gen = gen_sphere
    alg = algo_hc

    for ms in m:
        print(ms)
    if alg_params.minimising is False:
        max_value, max_index = max((x, (i, j))
                                   for i, row in enumerate(m)  # m_sd
                                   for j, x in enumerate(row))  # m_rate
    else:
        max_value, max_index = min((x, (i, j))
                                   for i, row in enumerate(m)
                                   for j, x in enumerate(row))
    max_index = (m_sds[max_index[0]], m_rates[max_index[1]])
    print()
    print('best fitness -', max_value)
    print('best params  - sd:', max_index[0], 'r:', max_index[1])

    fig_title = 'Parameter Tuning: Problem-' + gen.__name__[15:] + ', Algorithm-' + alg.__name__[16:] + ' (' + str(
        alg_params.generations) + ' gens)'
    f_name = 'param_tuning ' + gen.__name__[15:] + ' ' + alg.__name__[16:] + ' gens ' + str(alg_params.generations)
    # Transpose graph to get sd on x axis
    graphing.graph_parameter_tuning_matrix(m_sds, m_rates, m.T, save=True, title=fig_title, f_name=f_name,
                                           xy_labels=('Mutation SD', 'Mutation Rate'), m_ind=max_index, logo=False)


def build_parameter_matrix2(p1, p2, gen, alg, aps):
    p1 = np.array([0, 1, 2, 3])
    p2 = ([10, 11, 12, 13, 14, 15, 16, 17])
    all_sols = np.zeros((len(p1), len(p2)))

    for p1i, p1val in enumerate(p1):
        for p2i, p2val in enumerate(p2):
            all_sols[p1i][p2i] = 100
            print(all_sols)
            print()
    return all_sols


def build_parameter_matrix(p1, p2, gen, alg, aps, n):
    # p1 = sd, p2 = m_rate
    all_sols = np.zeros((len(p1), len(p2)))
    for p1i, p1val in enumerate(p1):  # sd
        print('Mutation sd:', p1val)
        for p2i, p2val in enumerate(p2):  # rate
            # print('Mutation sd:', p2val)
            aps.mutation_rate = p2val
            aps.mutation_sd = p1val

            s = run_n_times(gen, alg, aps, n, print_ind=False, print_alg=False)
            all_sols[p1i][p2i] = s
    # Returns rows as standard deviation changes
    #         columns as mutation rate changes
    return all_sols


if __name__ == '__main__':
    # -- Graph multiple Algorithm's performance on a specific problem generator --
    algos = [algo_ea, algo_saes, algo_saea, algo_hc, algo_rs]
    gen = gen_schwefel
    graph_multiple_algos_performance(algos, gen, 50)

    # -- Specify Algorithm Parameters --
    # aps = solver.AlgoParams(generations=500,
    #                         pop_size=20,
    #                         num_children=15,
    #                         mutation_rate=0.5,
    #                         mutation_sd=0.05,
    #                         crossover_rate=0.2,
    #                         minimising=True)
    # run_with_stats([gen_schwefel, gen_rastrigin], [algo_saea3], aps, 60)

    # -- Run simple optimisation --
    # s = solver.Solver(gen_schwefel, algo_saea, aps).solve()
    # print(s.fitness)

    # -- Run simple optimisation multiple times --
    # s = run_n_times(gen_schwefel, algo_saes, aps, 5, print_ind=True, print_alg=True)
    # s = run_n_times(gen_schwefel, algo_ea, aps, 5, print_ind=True, print_alg=True)

    # -- Graph Parameter Tuning Matrix --
    # m_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    # m_sds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1]
    # gen = gen_rastrigin
    # m = build_parameter_matrix(m_sds, m_rates, gen, alg, alg_params, num_runs)
    # plot_params_from_existing_matrix(None, None, None, None, None, None)
    # t_parameter_tuning()

    # -- Graph Algorithm performance --
    # gen = gen_rastrigin
    # algos = [algo_ea, algo_saes, algo_saea]
    # run_with_stats(gen, algos, aps, 20)
