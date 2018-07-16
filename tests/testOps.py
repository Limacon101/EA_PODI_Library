import unittest
from unittest.mock import patch, Mock, MagicMock
from core.solver import Trace
from core import ops, solver
import random
import copy
from algorithms import algo_ea


def mock_trunc_norm(mean, sd, lower=1e-12, upper=1):
    if mean - sd > lower:
        return mean - sd
    elif mean + sd < upper:
        return mean + sd
    else:
        return mean


def mock_create_solution_with_raw_trace(x):
    t = create_mock_trace(x)
    return t


def create_mock_trace(trace=None, es_params=None):
    if es_params is None:  # Assignment required here due to default arguments being immutable
        es_params = []
    if trace is None:
        trace = []
    t = Mock(spec=solver.Trace)
    t.trace = trace.copy()
    t.es_params = es_params
    return t


class TestOps(unittest.TestCase):

    def setUp(self):
        # AlgoParams container with default arguments
        aps = solver.AlgoParams()
        mock_solver = Mock(spec=solver.Solver)
        mock_solver.alg_params = aps
        mock_solver.create_solution_with_raw_trace.side_effect = mock_create_solution_with_raw_trace
        self.mock_solver = mock_solver

    def testTruncNorm(self):
        # Test with minimal s.d.
        n = ops.get_trunc_norm(0.5, 1e-10, 0, 1)
        self.assertAlmostEqual(n, 0.5)  # To 7 places

        # Test with bounded mean
        n = ops.get_trunc_norm(-1, 1e-10, 0, 1)
        self.assertAlmostEqual(n, 0)
        n = ops.get_trunc_norm(2, 1e-10, 0, 1)
        self.assertAlmostEqual(n, 1)

        # Test with invalid s.d.
        self.assertRaises(AssertionError, ops.get_trunc_norm, 0.5, -0.1, 0, 1)

    @patch('core.ops.get_trunc_norm', side_effect=mock_trunc_norm)
    def testMutateTraceGauss(self, mock_trunc_norm):
        random.seed(101)
        max_trace_len = 100

        # Simple check to ensure trace does not mutate with 0 mutation rate
        t1_val = [random.random() for _ in range(random.randint(1, max_trace_len))]
        t1 = create_mock_trace(t1_val)
        t1_res = ops.mutate_trace_gauss(self.mock_solver, t1, 0, 0.1)
        self.assertEqual(t1_res.trace, t1_val)

        # Check that trace mutates each value (using patched trunc_norm)
        t2_val = [random.random() for _ in range(random.randint(1, max_trace_len))]
        t2 = create_mock_trace(t2_val)
        t2_res = ops.mutate_trace_gauss(self.mock_solver, t2, 1, 0.1)
        for r1, r2 in zip(t2_res.trace, t2_val):
            self.assertNotEqual(r1, r2)
        self.assertNotEqual(t2_res.trace, t2_val)

    def testCrossoverOnePoint(self):
        random.seed(101)
        max_trace_len = 100

        # Test that the crossover point works for varying values
        for i in range(100):
            t1_val = [random.random() for _ in range(random.randint(1, max_trace_len))]
            t1 = create_mock_trace(t1_val)

            t2_val = [random.random() for _ in range(random.randint(1, max_trace_len))]
            t2 = create_mock_trace(t2_val)

            t = ops.crossover_one_point(self.mock_solver, t1, t2, xo_chance=1, point=max_trace_len+1)
            self.assertEqual(t.trace, t1_val)

            t = ops.crossover_one_point(self.mock_solver, t1, t2, xo_chance=1, point=0)
            self.assertEqual(t.trace, t2_val)

        # Test that crossover works at all split points
        for split in range(1, max_trace_len+1):
            t1_val = [random.random() for _ in range(max_trace_len)]
            t1 = create_mock_trace(t1_val)

            t2_val = [random.random() for _ in range(max_trace_len)]
            t2 = create_mock_trace(t2_val)

            r1 = ops.crossover_one_point(self.mock_solver, t1, t2, xo_chance=1, point=split)
            self.assertEqual(t1_val[:split], r1.trace[:split])
            self.assertEqual(t2_val[split:], r1.trace[split:])








if __name__ == '__main__':
    unittest.main()

