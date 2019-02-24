#include "ffnn/actf/ExponentialActivationFunction.hpp"

#include <math.h>


// Activation Function Interface implementation


double ExponentialActivationFunction::f(const double &in)
{
    return exp(in);
}


double ExponentialActivationFunction::f1d(const double &in)
{
    return exp(in);
}


double ExponentialActivationFunction::f2d(const double &in)
{
    return exp(in);
}


double ExponentialActivationFunction::f3d(const double &in)
{
    return exp(in);
}

void ExponentialActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    v = exp(in);
    v1d = flag_d1 ? v : 0.;
    v2d = flag_d2 ? v : 0.;
    v3d = flag_d3 ? v : 0.;
}
