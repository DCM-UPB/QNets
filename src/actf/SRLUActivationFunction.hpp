#ifndef SRLU_ACTIVATION_FUNCTION
#define SRLU_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>

// Smooth Rectified Linear Unit ( == ln(1+exp(x)) )
class SRLUActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy(){return new SRLUActivationFunction();}
    std::string getIdCode(){return "SRLU";}

    // input should be in the rage [-5 : 5] -> mu=0   sigma=10/sqrt(12)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 2.886751345948129;}

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};

#endif
