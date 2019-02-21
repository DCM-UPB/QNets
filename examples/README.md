# LEGEND OF THE EXAMPLES

Make sure the examples are compiled, by running `./build.sh` in the project root folder.
Execute an example by switching into one of the example folders and running `./run.sh`.
Note that the actual example executables reside inside the build/ folder, from project root.
Some examples might also contain a `plot.py` script to show a plot.
Run it after the executable has terminated, via `python plot.py` (requires matplotlib).


## Example 1

`ex1/`: construct and modify the geometry of a FFNN (i.e. number of layers and units)



## Example 2

`ex2/`: access and modify the beta of a FFNN (i.e. the variational parameters)



## Example 3

`ex3/`: manage the activation functions



## Example 4

`ex4/`: compute the output of a FFNN



## Example 5

`ex5/`: compute first and second derivative of the NN output in respect to input



## Example 6

`ex6/`: compute first derivative of the NN in respect to the variational parameters (betas)



## Example 7

`ex7/`: write files for plotting the values and its first and second derivatives of a NN:R->R (1-dimensional function)



## Example 8

`ex8/`: read FFNN structure from a file



## Example 9

`ex9/`: use NNTrainer(GSL) to make FFNN fit a gaussian



## Example 10

`ex10/`: use a FeatureMapLayer to fit more easily a gaussian of x-y-distance
