#include "TanSigmoidActivationFunction.hpp"

#include <math.h>

// Activation Function Interface implementation

double TanSigmoidActivationFunction::f(const double &in)
{
    return 2.0 / (1.0 + exp(-2.0*in)) - 1.0;
}


double TanSigmoidActivationFunction::f1d(const double &in)
{
    double expf = exp(-2.0*in);
    double quot = 2.0 / (1.0 + expf);

    return expf * quot * quot; // 4 * exp(-2*in) / (1 + exp(-2*in))^2
}


double TanSigmoidActivationFunction::f2d(const double &in)
{
    double expf = exp(-2.0*in);
    double quot = 2.0 / (1.0 + expf);
    double prod = expf * quot;

    return 2.0 * prod * quot * (prod - 1.0);  // 8 * exp(-2*in) / (1 + exp(-2*in))^2 * (2 * exp(-2*in) / (1 + exp(-2*in)) - 1)
}


double TanSigmoidActivationFunction::f3d(const double &in)
{
    double expf = exp(-2.0*in);
    double quot = 2.0 / (1.0 + expf);
    double prod = expf * quot;
    
    return 4.0 * prod * quot * (1.5 * prod * prod - 3.0 * prod + 1.0);  // 16 * exp(-2*in) / (1 + exp(-2*in))^2 * ( 6 * exp(-2*in) / (1 + exp(-2*in)) * (exp(-2*in) / (1 + exp(-2*in)) - 1) + 1)
}
