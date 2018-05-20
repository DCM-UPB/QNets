#ifndef SINE_ACTIVATION_FUNCTION
#define SINE_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class SineActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    SineActivationFunction(){}
    ~SineActivationFunction(){}

    std::string getIdCode(){return "sine";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);
};


#endif
