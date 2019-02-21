#ifndef TANSIGMOID_ACTIVATION_FUNCTION
#define TANSIGMOID_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class TanSigmoidActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy(){return new TanSigmoidActivationFunction();}
    std::string getIdCode(){return "TANS";}

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 0.577350269189626;}

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
