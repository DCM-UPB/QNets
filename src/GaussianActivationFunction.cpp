#include "GaussianActivationFunction.hpp"

#include <math.h>



// Activation Function Interface implementation

double GaussianActivationFunction::f(const double &in)
{
    return exp(-in*in);
}


double GaussianActivationFunction::f1d(const double &in)
{
    return -2.*in*exp(-in*in);
}


double GaussianActivationFunction::f2d(const double &in)
{
    return 2.*exp(-in*in)*( -1. + 2.*in*in );
}


double GaussianActivationFunction::f3d(const double &in)
{
    return 4.*in*exp(-in*in)*( 3. - 2.*in*in );
}
