#include "SELUActivationFunction.hpp"

#include <math.h>


// Activation Function Interface implementation

double SELUActivationFunction::f(const double &in)
{
    return in>0.0 ? _lambda*in : _alpha*exp(in) - _alpha;
}


double SELUActivationFunction::f1d(const double &in)
{
    return in>0.0 ? _lambda : _alpha*exp(in);
}


double SELUActivationFunction::f2d(const double &in)
{
    return in>0.0 ? 0.0 : _alpha*exp(in);
}


double SELUActivationFunction::f3d(const double &in)
{
    return in>0.0 ? 0.0 : _alpha*exp(in);
}
