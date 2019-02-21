#include "ffnn/actf/SineActivationFunction.hpp"

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

void SineActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    v = sin(in);
    v2d = flag_d2 ? -v : 0.0;

    if (flag_d1) {
        v1d = cos(in);
        v3d = flag_d3 ? -v1d : 0.0;
    }
    else {
        v1d = 0.0;
        v3d = flag_d3 ? -cos(in) : 0.0;
    }
}
