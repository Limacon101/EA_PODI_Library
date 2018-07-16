import unittest

from tests import testAlgos, testOps, testCore


def run_test_suite():
    """
    Run the imported test cases

    :return:
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests = [testAlgos, testOps, testCore]
    for test in tests:
        suite.addTests(loader.loadTestsFromModule(test))

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)


if __name__ == '__main__':
    run_test_suite()
