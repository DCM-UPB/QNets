#ifndef FFNN_ACTF_SINEACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_SINEACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>


class SineActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() override{return new SineActivationFunction();}
    std::string getIdCode() override{return "SIN";}

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() override{return 0.;}
    double getIdealInputSigma() override{return 0.577350269189626;}

    // full output is in the range [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getOutputMu(const double & /*inputMu*/ = 0., const double & /*inputSigma*/ = 1.) override{return 0.;}
    double getOutputSigma(const double & /*inputMu*/ = 0., const double & /*inputSigma*/ = 1.) override{return 1./sqrt(3);}

    // computation
    double f(const double &in) override;

    double f1d(const double &in) override;

    double f2d(const double &in) override;

    double f3d(const double &in) override;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) override;
};


#endif
