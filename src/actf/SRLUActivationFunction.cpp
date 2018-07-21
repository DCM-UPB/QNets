#include "SRLUActivationFunction.hpp"

#include <cmath>


// Activation Function Interface implementation


double SRLUActivationFunction::f(const double &in)
{
    return log1p(exp(in)); // log(1+exp(in))
}


double SRLUActivationFunction::f1d(const double &in)
{
    return exp(in)/(1.+exp(in));
}


double SRLUActivationFunction::f2d(const double &in)
{
    const double v1d = exp(in)/(1.+exp(in));
    return v1d - v1d*v1d;
}


double SRLUActivationFunction::f3d(const double &in)
{
    const double v1d = exp(in)/(1.+exp(in));
    const double v1dsq = v1d * v1d;
    return v1d - 3.*v1dsq + 2.*v1dsq*v1d;
}

void SRLUActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    const double expin = exp(in);
    v = log1p(expin); // log(1+exp(in))

    if (flag_d1 || flag_d2 || flag_d3) {
        const double v1dh = expin / (1.+expin);
        v1d = flag_d1 ? v1dh : 0.;
        v2d = flag_d2 ? v1dh - v1dh*v1dh : 0.;
        v3d = flag_d3 ? v1dh - 3.*v1dh*v1dh + 2.*v1dh*v1dh*v1dh : 0.;
    }
}
