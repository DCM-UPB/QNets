#include "ffnn/actf/SELUActivationFunction.hpp"
#include "ffnn/serial/StringCodeUtilities.hpp"

#include <math.h>
#include <string>
#include <vector>


std::string SELUActivationFunction::getParams()
{
    std::vector<std::string> paramCodes;
    paramCodes.push_back(composeParamCode("alpha", _alpha));
    paramCodes.push_back(composeParamCode("lambda", _lambda));
    return composeCodeList(paramCodes);
}

void SELUActivationFunction::setParams(const std::string &params)
{
    setParamValue(params, "alpha", _alpha);
    setParamValue(params, "lambda", _lambda);
}

// Activation Function Interface implementation

double SELUActivationFunction::f(const double &in)
{
    return in>0.0 ? _lambda*in : _alpha*exp(in) - _alpha;
}


double SELUActivationFunction::f1d(const double &in)
{
    return in>0.0 ? _lambda : _alpha*exp(in);
}


double SELUActivationFunction::f2d(const double &in)
{
    return in>0.0 ? 0.0 : _alpha*exp(in);
}


double SELUActivationFunction::f3d(const double &in)
{
    return in>0.0 ? 0.0 : _alpha*exp(in);
}

void SELUActivationFunction::fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1, const bool flag_d2, const bool flag_d3)
{
    if (in>0.0) {
        v = _lambda*in;
        v1d = flag_d1 ? _lambda : 0.0;
        v2d = 0.0;
        v3d = 0.0;
    }
    else {
        const double aexp = _alpha*exp(in);
        v = aexp - _alpha;
        v1d = flag_d1 ? aexp : 0.0;
        v2d = flag_d2 ? aexp : 0.0;
        v3d = flag_d3 ? aexp : 0.0;
    }
}
