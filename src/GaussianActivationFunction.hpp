#ifndef GAUSSIAN_ACTIVATION_FUNCTION
#define GAUSSIAN_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class GaussianActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    // getters
    ActivationFunctionInterface * getCopy(){return new GaussianActivationFunction();}
    std::string getIdCode(){return "GSS";}

    // input should be in the rage [-3 : 3]
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return ;}

    double getOutputMu(){return ;}
    double getOutputSigma(){return ;}

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
