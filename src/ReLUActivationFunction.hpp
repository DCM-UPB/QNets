#ifndef RELU_ACTIVATION_FUNCTION
#define RELU_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class ReLUActivationFunction: public ActivationFunctionInterface
{
protected:
    const double _alpha;

public:
    ReLUActivationFunction(const double alpha = 0.0): _alpha(alpha) {}
    ~ReLUActivationFunction(){}

    std::string getIdCode(){return "relu";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
