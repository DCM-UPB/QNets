# LEGEND OF THE EXAMPLES

Make sure the examples are compiled, by running `./build.sh` in the project root folder.
Execute an example by switching into one of the example folders and running `./run.sh`.
Note that the actual example executables reside inside the `build/examples/` folder under the project's root.
Some examples might also contain a `plot.py` script to show a plot. It gets called automatically
by `./run.sh` after the executable has terminated, but requires python with matplotlib.


## Basic Example

`ex_basic/`: construct and modify the geometry of a FFNN (i.e. number of layers and units)



## Network weights (beta)

`ex_beta/`: access and modify the beta of a FFNN (i.e. the variational parameters)



## Activation Functions

`ex_actf/`: manage the activation functions



## NN Propagation

`ex_propagate/`: compute the output of a FFNN



## Input Derivative

`ex_xderiv/`: compute first and second derivative of the NN output in respect to input



## Variational Derivative

`ex_vderiv/`: compute first derivative of the NN in respect to the variational parameters (betas)



## Printing and Plotting

`ex_plot/`: write files for plotting the values and its first and second derivatives of a NN:R->R (1-dimensional function)



## Load NN from file

`ex_loadfile/`: read FFNN structure from a file



## Fit NN function to data

`ex_fit/`: use NNTrainer(GSL) to make FFNN fit a gaussian



## Fit a NN with feature map layer

`ex_features/`: use a FeatureMapLayer to fit more easily a gaussian of x-y-distance
