#include "SELUActivationFunction.hpp"

#include <math.h>
#include <string>
#include <sstream>

std::string SELUActivationFunction::getParams()
{
    std::ostringstream oss;
    oss << "alpha ";
    oss << _alpha;
    oss << " lambda ";
    oss << _lambda;
    return oss.str();
}

void SELUActivationFunction::setParams(const std::string &params)
{
    std::istringstream iss(params);
    std::string word;
    while( iss >> word ) {
        if (word == "alpha") {
            iss >> _alpha;
        }
        else if (word == "lambda") {
            iss >> _lambda;
        }
    }
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
