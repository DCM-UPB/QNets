# LEGEND OF THE UNIT TESTS

Use `./run.sh` inside the test directory to run the check program and unit tests
with valgrind or use `make test` inside the build directory, to run unit tests without valgrind.


## Unit Test 1

`ut1/`: check that the FFNN derivatives are computed correctly


## Unit Test 2

`ut2/`: check that storing and retrieving a FFNN with a file works properly


## Unit Test 3

`ut3/`: check that the cloning constructor of FFNN works properly


## Unit Test 4

`ut4/`: check that the compute and propagate work properly


## Unit Test 5

`ut5/`: check that the PolyNet activation functions derivatives are correct


## Unit Test 6

`ut6/`: check the utility functions of the string code system


## Unit Test 7

`ut7/`: check the SmartBetaGenerator functions


## Unit Test 8

`ut8/`: check that the GSL residual jacobians are correct


## Unit Test 9

`ut9/`: check that the trainers find perfect fits for a target function resembling a NN


## Unit Test 10

`ut10/`: check the derivatives and file storing when feature maps are used


## Unit Test 11

`ut11/`: check that the TemplNet activation functions derivatives are correct


## Unit Test 12

`ut12/`: check the results of basic TemplNet methods


## Unit Test 13

`ut13/`: check TemplNet propagation by comparing against the already checked PolyNet