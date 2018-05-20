#include "SineActivationFunction.hpp"

#include <math.h>

// Activation Function Interface implementation

double SineActivationFunction::f(const double &in)
{
    return sin(in);
}


double SineActivationFunction::f1d(const double &in)
{
    return cos(in);
}


double SineActivationFunction::f2d(const double &in)
{
    return -sin(in);
}


double SineActivationFunction::f3d(const double &in)
{
    return -cos(in);
}
