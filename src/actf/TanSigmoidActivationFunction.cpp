#include "ffnn/actf/TanSigmoidActivationFunction.hpp"

#include <cmath>

// Activation Function Interface implementation


double TanSigmoidActivationFunction::f(const double &in)
{
    return 2.0 / (1.0 + exp(-2.0*in)) - 1.0;
}


double TanSigmoidActivationFunction::f1d(const double &in)
{
    const double expf = exp(-2.0*in);
    const double quot = 2.0 / (1.0 + expf);

    return expf * quot * quot; // 4 * exp(-2*in) / (1 + exp(-2*in))^2
}


double TanSigmoidActivationFunction::f2d(const double &in)
{
    const double expf = exp(-2.0*in);
    const double quot = 2.0 / (1.0 + expf);
    const double prod = expf * quot;

    return 2.0 * prod * quot * (prod - 1.0);  // 8 * exp(-2*in) / (1 + exp(-2*in))^2 * (2 * exp(-2*in) / (1 + exp(-2*in)) - 1)
}


double TanSigmoidActivationFunction::f3d(const double &in)
{
    const double expf = exp(-2.0*in);
    const double quot = 2.0 / (1.0 + expf);
    const double prod = expf * quot;

    return 4.0 * prod * quot * (1.5 * prod * prod - 3.0 * prod + 1.0);  // 16 * exp(-2*in) / (1 + exp(-2*in))^2 * ( 6 * exp(-2*in) / (1 + exp(-2*in)) * (exp(-2*in) / (1 + exp(-2*in)) - 1) + 1)
}

void TanSigmoidActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    const double expf = exp(-2.0*in);
    const double quot = 2.0 / (1.0 + expf);
    const double prod = expf * quot;

    v = quot - 1.0;

    if (flag_d1) {
        v1d = prod * quot;
        v2d = flag_d2 ? 2.0 * v1d * (prod - 1.0) : 0.;
        v3d = flag_d3 ? 4.0 * v1d * (1.5 * prod * prod - 3.0 * prod + 1.0) : 0.;
    }
    else{
        v1d = 0.;
        v2d = flag_d2 ? 2.0 * prod * quot * (prod - 1.0) : 0.;
        v3d = flag_d3 ? 4.0 * prod * quot * (1.5 * prod * prod - 3.0 * prod + 1.0) : 0.;
    }
}
