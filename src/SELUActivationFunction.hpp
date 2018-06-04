#ifndef SELU_ACTIVATION_FUNCTION
#define SELU_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class SELUActivationFunction: public ActivationFunctionInterface
{
protected:
    const double _alpha, _lambda;

public:
    SELUActivationFunction(const double alpha = 1.6733, const double lambda = 1.0507): _alpha(alpha), _lambda(lambda){}
    ~SELUActivationFunction(){}

    std::string getIdCode(){return "selu";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
