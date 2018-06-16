#ifndef SINE_ACTIVATION_FUNCTION
#define SINE_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class SineActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    // getters
    ActivationFunctionInterface * getCopy(){return new SineActivationFunction();}
    std::string getIdCode(){return "SIN";}

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
