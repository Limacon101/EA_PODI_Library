import random
from core import ops

from algorithms import algo_rs, algo_ea, algo_hc
from generators import gen_onemax, gen_gp, gen_sphere


class Solver(object):

    def __init__(self, generator, algorithm, alg_params=None):
        """
        Instantiate a Solver before starting an optimisation

        :param generator:           Problem generator
        :param algorithm:           Optimisation algorithm
        :param alg_params:          Algorithm parameters
        """
        if alg_params is None:
            alg_params = AlgoParams()
        self.alg_params = alg_params
        alg_params.validate()

        self.generator = generator
        Solver.validate_generator(self.generator)
        self.algorithm = algorithm
        Solver.validate_algorithm(self.algorithm)
        self.stats = Stats(alg_params.generations)

    @staticmethod
    def validate_generator(generator):
        if not hasattr(generator, 'create_solution'):
            print('Invalid Generator')
            raise AttributeError('Invalid Generator.\nGenerator must contain a function with signature: '
                                 'create_solution(t).\n'
                                 'This method generates a random candidate solution to a problem, '
                                 'where t is a dependency injected used to generate random numbers')
        if not hasattr(generator, 'fitness'):
            raise AttributeError('Invalid Generator.\nGenerator must contain a function with signature: '
                                 'fitness(sol)\n'
                                 'This method calculates the fitness of a candidate solution, '
                                 'where sol is a candidate solution ')
        return True

    @staticmethod
    def validate_algorithm(algorithm):
        if not hasattr(algorithm, 'run'):
            print('Invalid Algorithm')
            raise AttributeError('Invalid Algorithm.\nAlgorithm must contain a function with signature: '
                                 'run(solver [, alg_operators]).\n'
                                 'This method runs the optimisation algorithm, where solver is a reference'
                                 'to the Solver object that initiated the optimisation process')
        return True

    def solve(self):
        """
        Optimise the problem

        :return:                Trace object containing a solution
        """
        return self.algorithm.run(self, self.alg_params)

    def create_solution(self, es_params=None):
        """
        Create a new solution

        :param es_params:       Additional parameters
        :return:                Trace object
        """
        t = Trace()
        sol = self.generator.create_solution(t)
        t.solution = sol
        t.new = False
        t.fitness = self.generator.fitness(t.solution)
        if es_params is not None:
            t.es_params = es_params
        return t

    def create_solution_with_trace(self, t):
        """
        Create a solution using an existing Trace object

        :param t:               Trace object
        :return:                Trace object
        """
        sol = self.generator.create_solution(t)
        t.solution = sol
        t.count = -1
        t.fitness = self.generator.fitness(t.solution)
        return t

    def create_solution_with_raw_trace(self, trace):
        """
        Create a solution using a list of random numbers

        :param trace:           List of numbers
        :return:                Trace object
        """
        t = Trace()
        t.new = False
        t.trace = trace
        sol = self.generator.create_solution(t)

        # Trims list if random numbers extend (otherwise some mutations have no effect)
        if t.count < len(t.trace):
            t.trace = t.trace[-(len(t.trace) - t.count - 1):]
        t.solution = sol
        t.count = -1
        t.fitness = self.generator.fitness(t.solution)
        return t


class Stats(object):

    def __init__(self, num_gens):
        self.num_gens = num_gens
        self.best = []
        self.gen_best = []
        self.gen_average = []

    def update_stats(self, best, gen_best=None, gen_av_fitness=None):
        self.best.append(best.fitness)
        if gen_best is not None:
            self.gen_best.append(gen_best.fitness)
        if gen_av_fitness is not None:
            self.gen_average.append(gen_av_fitness)

    def get_stats(self):
        """
        Gets generation bests, current generation bests and current generation
        averages

        :return:    A tuple with 3 lists of fitnesses
        """
        self.validate_stats()
        return self.best, self.gen_best, self.gen_average

    def get_generational_overall_bests(self):
        """
        For each generation, gets the best solution fitness found at this point
        (This value should always be getting 'better')

        :return:    A list of fitnesses
        """
        self.validate_stats()
        return self.best

    def get_generational_bests(self):
        """
        For each generation, gets the best solution found during this generation
        (This value may get worse for non-elitist evolutionary algorithms)

        :return:    A list of fitnesses
        """
        self.validate_stats()
        return self.gen_best

    def get_generational_averages(self):
        """
        For each generation, gets the generational average
        (This value may get worse for non-elitist evolutionary algorithms)

        :return:    A list of fitnesses
        """
        self.validate_stats()
        return self.gen_average

    def validate_stats(self):
        if len(self.gen_best) > 0:
            assert len(self.gen_best) == len(self.best)
        if len(self.gen_average) > 0:
            assert len(self.gen_average) == len(self.best)


class AlgoParams:

    def __init__(self, select=None, mutate=None, crossover=None,
                 generations=100, pop_size=30, num_children=None,
                 mutation_rate=0.1, mutation_sd=0.1, crossover_rate=1,
                 minimising=True, verbose=-1):
        """
        Arguments passed to an algorithm.
        An algorithm can choose to use these parameters or set its own
         (E.g. A SAEA may dynamically set the mutation_rate based on a solution's parameters)

        Note: Functional arguments set as default None -- due to Python's use of mutable default arguments

        :param select:          Selection operator
        :param mutate:          Mutation operator
        :param crossover:       Crossover operator
        :param generations:     Number of generations (for iterative algorithms)
        :param pop_size:        Population Size (for population-based algorithms)
        :param mutation_rate:   Mutation rate
        :param crossover_rate:  Crossover rate
        :param minimising:      Whether algorithm should be minimising or maximising fitness
        :param verbose:         -1: No printout     0: Detailed (runthrough)    1-N: Print generation stats every n gens
        """
        if select is None:
            select = ops.select_tournament
        if mutate is None:
            mutate = ops.mutate_trace_gauss
        if crossover is None:
            crossover = ops.crossover_one_point
        self.select = select
        self.mutate = mutate
        self.crossover = crossover

        self.generations = generations
        self.pop_size = pop_size
        self.num_children = self.pop_size if num_children is None else num_children
        self.mutation_rate = mutation_rate
        self.mutation_sd = mutation_sd
        self.crossover_rate = crossover_rate

        self.minimising = minimising
        self.verbose = verbose

    def __str__(self):
        s = ''
        s += 'selection op: ' + str(self.select.__name__) + '\n'
        s += 'mutation op: ' + str(self.mutate.__name__) + '\n'
        s += 'crossover op: ' + str(self.crossover.__name__) + '\n'
        s += 'generations: ' + str(self.generations) + '\n'
        s += 'pop_size: ' + str(self.pop_size) + '\n'
        s += 'num_children: ' + str(self.num_children) + '\n'
        s += 'mutation_rate: ' + str(self.mutation_rate) + '\n'
        s += 'mutation_sd: ' + str(self.mutation_sd) + '\n'
        s += 'crossover_rate: ' + str(self.crossover_rate) + '\n'
        s += 'minimising: ' + str(self.minimising) + '\n'
        return s

    def validate(self):
        # Roulette selection cannot be used when minimising
        if self.select is ops.select_roulette:
            assert self.minimising is False

        assert self.generations > 0
        assert self.pop_size > 0
        assert 0 <= self.mutation_rate <= 1
        assert 0 <= self.mutation_sd
        assert self.verbose >= -1
        assert self.num_children <= self.pop_size  # For Select_best_n method


class Trace(object):
    """
    @:param es_params       Additional parameters (e.g. parameters for self-adaptation)
    @:param new:            True if Trace created for the first time
    @:param trace:          List holding trace values
    @:param solution        Solution calculated from the trace
    @:param fitness         Fitness associated with the solution
    @:param count           During recall for a generator -- the position reached in the trace (reset externally)
    """
    def __init__(self):
        self.sa_params = []
        self.es_params = []
        self.new = True

        self.trace = []
        self.solution = 0
        self.fitness = 0

        self.count = -1

    def __str__(self):
        return 'S: ' + str(self.solution) + '  f: ' + str(self.fitness)

    def random(self):
        """
        Returns a random number float between 0 and 1 and adds it to the trace

        :return:            Random real number 0..1
        """
        return self.add_to_trace(random.random())

    def randrange(self, start, stop):
        """
        Returns a random integer from range start to stop

        :param start:       Range start
        :param stop:        Range stop
        :return:            Random integer
        """
        if start >= stop:
            raise ValueError("stop must be larger than start")
        if not isinstance(start, int) or not isinstance(stop, int):
            raise TypeError("Inputs must be integers")
        r = self.random()
        return int(r * (stop - start) + start)

    def randint(self, start, stop):
        """
        Returns a random integer R from range start to stop such that start <= R <= stop

        :param start:       Range start
        :param stop:        Range stop
        :return:            Random integer
        """
        return self.randrange(start, stop + 1)

    def add_to_trace(self, r):
        """
        If the Trace is new, add random number to trace (solution creation).
        Otherwise get the next number from the existing trace (solution reconstruction).

        :param r:           Random real number 0..1
        :return:            'Random number from trace
        """
        if self.new:
            self.trace.append(r)
            return r
        else:
            return self.from_seed(r)

    def from_seed(self, r):
        """
        Get the next number from the trace. If the trace it too short add the random number supplied
        to the trace

        :param r:           Random real number 0..1
        :return:            'Random' number from trace
        """
        self.count += 1
        if self.count < len(self.trace):
            # print('r', self.trace[self.count])
            return self.trace[self.count]
        elif self.count == len(self.trace):
            # r = random.random()
            self.trace.append(r)
            return r
        else:
            raise ValueError('Count exceeded trace length')

    def print_trace(self):
        f_list = ['%.2f' % elem for elem in self.trace]
        print(f_list)


if __name__ == '__main__':
    gens = [gen_onemax, gen_sphere, gen_gp]
    algs = [algo_hc, algo_rs, algo_ea]

    for gen in gens:
        for alg in algs:
            print('\n\n', alg.__name__, ' --- ', gen.__name__)
            my_ops = AlgoParams(select=ops.select_tournament,
                                crossover=ops.crossover_one_point,
                                mutate=ops.mutate_trace,
                                minimising=True)
            s = Solver(gen, alg, my_ops).solve()
            print(s)

    # gen = gen_onemax
    # alg = algo_hc
    # s = Solver(gen, alg).solve()
    # print(s)

    # t = Trace(Representation.REAL)
    # print(t.randint(0, 3))
    # trace = t.trace
    # print(trace)

    # print("Calculating from core...")
    # gen = gen_sphere
    # alg = algo_hc
    # s = Solver(gen, alg)
    # print(s.representation)
    # print(s)
