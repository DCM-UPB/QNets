#ifndef FFNN_ACTF_GAUSSIANACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_GAUSSIANACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>


class GaussianActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() final { return new GaussianActivationFunction(); }
    std::string getIdCode() final { return "GSS"; }

    // input should be in the rage [-3 : 3]
    double getIdealInputMu() final { return 0.; }
    double getIdealInputSigma() final { return 1.732050807568877; }

    // full output is in the range [0 : 1] -> mu=0.5   sigma=1/sqrt(12)
    double getOutputMu(const double & /*inputMu*/ = 0., const double & /*inputSigma*/ = 1.) final { return 0.5; }
    double getOutputSigma(const double & /*inputMu*/ = 0., const double & /*inputSigma*/ = 1.) final { return 1./sqrt(12); }

    // computation
    double f(const double &in) final;

    double f1d(const double &in) final;

    double f2d(const double &in) final;

    double f3d(const double &in) final;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) final;
};


#endif
