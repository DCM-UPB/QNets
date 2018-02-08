#ifndef GAUSSIAN_ACTIVATION_FUNCTION
#define GAUSSIAN_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class GaussianActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    GaussianActivationFunction(){}
    ~GaussianActivationFunction(){}

    std::string getIdCode(){return "gss";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);
};


#endif
