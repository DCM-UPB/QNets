#include "ReLUActivationFunction.hpp"


// Activation Function Interface implementation

double ReLUActivationFunction::f(const double &in)
{
    return in>0.0 ? in : _alpha*in;
}


double ReLUActivationFunction::f1d(const double &in)
{
    return in>0.0 ? 1.0 : _alpha;
}


double ReLUActivationFunction::f2d(const double &in)
{
    return 0.0;
}


double ReLUActivationFunction::f3d(const double &in)
{
    return 0.0;
}

void ReLUActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    if (in>0.0) {
        v = in;
        v1d = flag_d1 ? 1.0 : 0.0;
    }
    else {
        v = _alpha*in;
        v1d = flag_d1 ? _alpha : 0.0;
    }
    v2d = 0.0;
    v3d = 0.0;
}
