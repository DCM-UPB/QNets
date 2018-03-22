#ifndef LOGISTIC_ACTIVATION_FUNCTION
#define LOGISTIC_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>

class LogisticActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    // getters
    ActivationFunctionInterface * getCopy(){return new LogisticActivationFunction();}
    std::string getIdCode(){return "LGS";}

    // input should be in the rage [-5 : 5] -> mu=0   sigma=sqrt(3)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 2.886751345948129;}

    // output is in the range [0 : 1] -> mu=0.5   sigma=1/(2*sqrt(3))
    double getOutputMu(){return 0.5;}
    double getOutputSigma(){return 0.288675134594813;}

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
