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

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 0.577350269189626;}

    // full output is in the range [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getOutputMu(const double &inputMu = 0., const double &inputSigma = 1.){return 0.;}
    double getOutputSigma(const double &inputMu = 0., const double &inputSigma = 1.){return 1./sqrt(3);}

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
