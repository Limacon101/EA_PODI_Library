import unittest
import random
from core.solver import Trace


class TestTrace(unittest.TestCase):

    def setUp(self):
        self.trace = Trace()

    def testConstruction(self):
        """
        Test Trace construction
        :return:
        """
        # t = Trace()
        self.assertEqual(self.trace.solution, 0)
        self.assertEqual(self.trace.fitness, 0)
        self.assertEqual(self.trace.new, True)
        self.assertEqual(self.trace.count, -1)
        self.assertEqual(self.trace.trace, [])

    def testAddToNewTrace(self):
        """
        Adding new number to new trace
        :return:
        """
        self.assertEqual(self.trace.new, True)
        r = 0.5  # Number to be added
        self.trace.add_to_trace(r)
        self.assertEqual(self.trace.trace, [0.5])
        r = 0
        self.trace.add_to_trace(r)
        self.assertEqual(self.trace.trace, [0.5, 0])

    def testFromSeed(self):
        """
        Tests getting numbers from existing (seeded) trace
        This test requires one extra number (that is not provided in the trace).
        :return:
        """
        random.seed(101)
        self.trace.new = False
        self.trace.trace = [0.1, 0.2, 0.3]
        recalled_values = []
        for i in range(4):
            recalled_values.append(self.trace.from_seed(random.random()))
        self.assertEqual(self.trace.trace, [0.1, 0.2, 0.3, 0.9239764016767943])
        self.assertEqual(self.trace.count, 3)

    def testRandom(self):
        """
        Test getting random number from *new* Trace (uses add_to_trace())
        This number should be stored in the trace
        :return:
        """
        random.seed(101)
        r = self.trace.random()
        self.assertAlmostEqual(r, 0.5811521325045647)
        self.assertAlmostEqual(self.trace.trace[0], 0.5811521325045647)

    def testRandRange(self):
        """
        Test getting random number from *new* Trace (uses add_to_trace())
        This number should be stored in the trace
        :return:
        """
        random.seed(101)
        r = self.trace.randrange(5, 10)
        self.assertEqual(self.trace.trace[0], 0.5811521325045647)
        self.assertEqual(r, 7)
        # Input: start >= stop:
        self.assertRaises(ValueError, self.trace.randrange, 5, 3)
        self.assertRaises(ValueError, self.trace.randrange, 5, 5)

        # Input: incorrect type:
        self.assertRaises(TypeError, self.trace.randrange, 5, "a string")
        self.assertRaises(TypeError, self.trace.randrange, None, 3)










if __name__ == '__main__':
    unittest.main()

