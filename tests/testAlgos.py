import unittest
from unittest.mock import patch, Mock, MagicMock
from core.solver import Trace, Solver, AlgoParams
from core import ops, solver
import random
import copy
from algorithms import algo_ea, algo_hc, algo_rs, algo_saea, algo_saes
from generators import gen_gp, gen_onemax, gen_rastrigin, gen_schwefel, gen_sphere


class TestAlgorithms(unittest.TestCase):

    def testAllAlgorithms(self):
        """
        Test all algorithms with all generators
        """
        algorithms = [algo_ea, algo_hc, algo_rs, algo_saea, algo_saes]
        gens = [gen_gp, gen_onemax, gen_rastrigin, gen_schwefel, gen_sphere]
        print('\nTesting algorithms:')
        for alg in algorithms:
            print('', alg.__name__)
            for gen in gens:
                aps = AlgoParams(generations=5, pop_size=5, num_children=3)
                s = Solver(gen, alg, aps).solve()
                self.assertIsNotNone(s.solution)
                self.assertIsNotNone(s.trace)


if __name__ == '__main__':
    unittest.main()