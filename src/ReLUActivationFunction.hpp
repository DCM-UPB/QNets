#ifndef RELU_ACTIVATION_FUNCTION
#define RELU_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class ReLUActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    ReLUActivationFunction(){}
    ~ReLUActivationFunction(){}

    std::string getIdCode(){return "relu";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);
};


#endif
