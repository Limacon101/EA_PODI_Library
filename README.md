# PODI Library
An Evolutionary Algorithm library utilising a Program Optimisation with Dependency Injection approach. This allows for problems to be independently specified from the algorithms used to optimise them. The library can compare the performance of a range of different algorithms (standard EA, self-adaptive EA, self-adaptive evolutionary strategy, hill-climber, random search) on a variety of problems (OneMax, Sphere, Schwefel's, Rastrigin's, Symbolic regression). The library is designed to be able to be easily extended to include new problems, algorithms and operators.

### Prerequisites
Python packages:
numpy
seaborn
matplotlib


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
