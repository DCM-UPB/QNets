# REPORT FOR EXAMPLE 10

This example is originally a demonstration of neural network training via GSL's fit routines.
But besides this it was also a vehicle to test how data normalization can improve our fitting
and how the fit results vary with number of units/layers.

## Data normalization

If you look at activation functions like the logistic function, you will see that they can roughly
be divided into 3 parts: a 'linear' part, a non-linear part with still large gradients and then
the remaining area of saturation, with small gradient. To prevent gradient signals from becoming too
small, we want to avoid the saturation, but still include the non-linear part. For the logistic function
this means something like the interval [a=-4, b=4].

Now remember that the input to a neuron's actf is the weighted sum of incoming connections. Assuming we
can control the statistical distribution of those incoming signals, like on the input side, we get for
signals of mean 0 and variance 1 and a weight distribution with variance Var(beta):

Var(sum) = sum_n(Var(beta) * Var(1)) = n Var(beta)

Sigma(sum) = sqrt(Var(sum)) = sqrt(n) Sigma(beta)

If we assume uniform distributions for sum and beta, we can calculate the interval bounds directly:

(b-a) = sqrt(12) * Sigma(sum) = sqrt(12) * sqrt(n) * (b_beta - a_beta) / sqrt(12) = sqrt(n) * (b_beta-a_beta)

where the sqrt(12) comes from the sigma of a flat distribution (sigma of a flat distribution in the range [a, b] = (b-a)/sqrt(12)), and where a_beta and b_beta are the boundaries for the interval from which the betas are drawn (i.e. the betas will be randomly sampled from [a_beta, b_beta])

In the symmetric case with b=-a=4 we get for b_beta:

b_beta = 4 / sqrt(n)

So concluding, if we can assume data to be normalized to mean 0 and variance 1, we can draw the betas
from a uniform distribution in [-b_beta, b_beta] to achieve logistic function x's mostly in [-4,4].


IMPORTANT: there are some limitations on what has beed said here
1 - the range [-4, 4] is optimal for the logistic function, not for all the activation functions
2 - the output of a logistic function does not have mean 0 and variance 1. This implies that this way of setting the betas cannot be extended to the layers following the first hidden layer



## Fitting results&discussion

To compare the performance for fitting a gaussian, several neural networks with various numbers units
and layers have been fitted (best out of 100 fits each) to the same gaussian data, by minimizing the
standard mean squared error.
The configurations were one layer with 5, 10, 15, 20 units and two layers with 5/5, 10/5, 5/10 and 10/10 units.
Plots comparing the function and derivative values as well as the overall root mean squared error can
be found in the 'plots' folder.

Overall, one can say that all configurations except the 5_0 yield visually excellent fits, even when looking
at the first derivative. The same does not hold for the second derivative, which is obviously not correctly
reproduced by any of the fitted neural networks.
Now, the plots of the root mean squared errors reveal a clear hierarchy (even for the second derivative),
with the 10_10 configuration coming out as best fit overall.
While the 5_0 and 10_10 are on their respective ends of the spectrum, where they were expected to be, at
least for the derivatives the hierarchy does not follow the number of units / layers strictly.
Since the number of fits from which the best fits were chosen, this is unlikely to be explained by
statistical error alone. Still, the exact reasons for these deviations from expectation remain unclear.

Nevertheless, it can be concluded that even with one hidden layer and only 10 units a gaussian and its
first derivative are approximated reasonably well, even though the fitting cost function did not include
the derivative. The second derivative seems to be a lot more challenging to reproduce though.
We also learn that in general the fits get better with more units/layers, but that this is not strictly the case.
