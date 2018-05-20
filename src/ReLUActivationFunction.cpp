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
