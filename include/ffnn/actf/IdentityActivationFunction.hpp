#ifndef FFNN_ACTF_IDENTITYACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_IDENTITYACTIVATIONFUNCTION_HPP


#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>

class IdentityActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() override { return new IdentityActivationFunction(); }
    std::string getIdCode() override { return "ID"; }

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() override { return 0.; }
    double getIdealInputSigma() override { return 0.577350269189626; }

    // output is identical to the input
    double getOutputMu(const double &inputMu = 0., const double & /*inputSigma*/ = 1.) override { return inputMu; }
    double getOutputSigma(const double & /*inputMu*/ = 0., const double &inputSigma = 1.) override { return inputSigma; }

    // computation
    double f(const double &in) override { return in; }
    double f1d(const double & /*in*/) override { return 1.; }
    double f2d(const double & /*in*/) override { return 0.; }
    double f3d(const double & /*in*/) override { return 0.; }

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool  /*flag_d2*/ = false, const bool  /*flag_d3*/ = false) override
    {
        v = in;
        v1d = flag_d1 ? 1. : 0.;
        v2d = 0.;
        v3d = 0.;
    }
};


#endif
