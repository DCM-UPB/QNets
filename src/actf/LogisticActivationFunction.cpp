#include "qnets/actf/LogisticActivationFunction.hpp"


// Activation Function Interface implementation


double LogisticActivationFunction::f(const double &in)
{
    return 1./(1. + exp(-in));
}


double LogisticActivationFunction::f1d(const double &in)
{
    const double f = this->f(in);
    return f*(1. - f);
}


double LogisticActivationFunction::f2d(const double &in)
{
    const double f = this->f(in);
    return f*(1. - f)*(1. - 2.*f);
}


double LogisticActivationFunction::f3d(const double &in)
{
    const double f = this->f(in);
    return f*(1. - f)*(1. - 6.*f + 6*f*f);
}

void LogisticActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    v = 1./(1. + exp(-in));

    if (flag_d1) {
        v1d = v*(1. - v);
        v2d = flag_d2 ? v1d*(1. - 2.*v) : 0.;
        v3d = flag_d3 ? v1d*(1. - 6.*v + 6.*v*v) : 0.;
    }
    else {
        v1d = 0.;
        v2d = flag_d2 ? v*(1. - v)*(1. - 2.*v) : 0.;
        v3d = flag_d3 ? v*(1. - v)*(1. - 6.*v + 6.*v*v) : 0.;
    }
}
