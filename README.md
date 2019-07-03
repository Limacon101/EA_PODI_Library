# Library In a Nutshell
This library can optimise any problem that can be expressed by providing two functions:
1. A function that generates a random solution in the problem's solution landscape
2. A function that evaluates a solution's fitness (its optimality)

A range of algorithms are provided: standard EA, self-adaptive EA, self-adaptive evolutionary strategy, hill-climber and random search.
The performance of these algorithms can be tested on a variety of different sample problems: OneMax, Sphere, Schwefel's, Rastrigin's, Symbolic regression.
The library has been designed to be easily extended to include new problems, algorithms and genetic operators.

The magic that allows different problems with vastly different solution representations can be found in the [core](https://github.com/Limacon101/EA_PODI_Library/blob/master/core/solver.py) module.

## Project Synopsis
Evolutionary algorithms are an effective approach for approximating hard optimisation problems. However, using an evolutionary algorithm for multiple problems or even problem instances requires both a time-consuming parameter tuning process and adaptation of the optimisation algorithm. This hybrid software-engineering and research project explores two approaches to addressing these problems: PODI and self-adaptation. Program Optimisation with Dependency Injection (PODI) is used for its ‘universal’ representation. Instead of evolving solutions, PODI evolves the program trace used to generate solutions. This means that all problems can be encoded with the same representation, allowing the system’s optimisation algorithms to be independent from solution generation. Parameter self-adaptation is then explored as a method for reducing the time-consuming algorithm parametertuning process. The project aims to find out whether using self-adaptation in a PODI library can be used as a competitive alternative to a standard evolutionary algorithm, potentially improving on the standard evolutionary algorithm’s performance and reducing the parameter tuning process. The experimental results show that parameter self-adaptation can be beneficial, with the exception of use on problems with a highly rugged fitness landscape.


## Prerequisites
Python packages:
* numpy
* seaborn
* matplotlib


## Sample run
The following code runs an optimisation for the Schwefel problem using a standard Evolutionary algorithm:
```python
from core.solver import Solver
from algorithms import algo_ea
from generators import gen_schwefel

s = Solver(gen_schwefel, algo_ea).solve()
```

Algorithm parameters and operators can be adjusted by passing the solver an AlgoParams object, e.g.:
```python
from core.solver import Solver
from algorithms import algo_ea
from generators import gen_schwefel

aps = solver.AlgoParams(generations=500,
                        pop_size=20,
                        num_children=15,
                        mutation_rate=0.5,
                        mutation_sd=0.05,
                        crossover_rate=0.2,
                        minimising=True)

s = Solver(gen_schwefel, algo_ea, aps).solve()
```

Examples of how to output visualisations of alogithmic performance comparisons are shown in core/sample_runs.py. Example visualisations include:
![Algorithm Performance Comparison](https://github.com/Limacon101/EA_PODI_Library/blob/master/results/alg_comparison/algorithm_progress.PNG)


## Sample optimisations (and visualisations of results)
Individual examples found in:
core/sample_runs.py


## Running Tests
Run all automated tests for system:
python/tests/runner.py

Individual test-cases can be found in modules:
tests/testAlgos.py
tests/testCore.py
tests/testOps.py


## Author
James Bache


## Licence
This project is licenced under the Apache License, Version 2.0 - see the [LICENCE.txt](LICENCE.txt) file for details
