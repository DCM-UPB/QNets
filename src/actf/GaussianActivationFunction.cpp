#include "qnets/actf/GaussianActivationFunction.hpp"



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
    return 2.*exp(-in*in)*(-1. + 2.*in*in);
}


double GaussianActivationFunction::f3d(const double &in)
{
    return 4.*in*exp(-in*in)*(3. - 2.*in*in);
}

void GaussianActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    const double in2 = in*in;

    v = exp(-in2);
    v1d = flag_d1 ? -2.0*in*v : 0.0;
    v2d = flag_d2 ? 4.0*v*(-0.5 + in2) : 0.0;
    v3d = flag_d3 ? 8.0*in*v*(1.5 - in2) : 0.0;
}
