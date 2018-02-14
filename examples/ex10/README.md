# REPORT FOR EXAMPLE 10

This example is originally a demonstration of neural network training via GSL's fit routines.
But besides this it was also a vehicle to test how data normalization can improve our fitting
and how the fit results vary with number of units/layers.

## Data normalization

If you look at activation functions like the logistic function, you will see that they can roughly
be divided into 3 parts: a 'linear' part, a non-linear part with still large gradients and then
the remaining area of saturation, with small gradient. To prevent gradient signals from becoming too 
small, we want to avoid the saturation, but still include the non-linear part. For the logistic function
this means something like the interval [-4,4].

Now remember that the input to a neuron's actf is the weighted sum of incoming connections. Assuming we
can control the statistical distribution of those incoming signals, like on the input side, we get for
signals of mean 0 and variance 1 and a weight distribution with variance Var(beta):

Var(sum) = sum_n(Var(beta) * Var(1)) = n Var(beta)
Sigma(sum) = sqrt(Var(sum)) = sqrt(n) Sigma(beta)

If we assume uniform distributions for sum and beta, we can calculate the interval bounds directly:

(b-a) = sqrt(12) * Sigma(sum) = sqrt(12) * sqrt(n) * (b_beta - a_beta) / sqrt(12) = sqrt(n) * (b_beta-a_beta)

In the symmetric case with b=-a=4 we get for a_beta:

b_beta = 4 / sqrt(n)

So concluding, if we can assume data to be normalized to mean 0 and variance 1, we can draw the betas from a uniform
distribution in [-b_beta, b_beta] to achieve logistic function x's mostly in [-4,4].

## Fitting results&discussion
