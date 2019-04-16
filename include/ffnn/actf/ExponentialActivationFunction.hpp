#ifndef FFNN_ACTF_EXPONENTIALACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_EXPONENTIALACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>

class ExponentialActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() override { return new ExponentialActivationFunction(); }
    std::string getIdCode() override { return "EXP"; }

    // input should be in the range [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() override { return 0.; }
    double getIdealInputSigma() override { return 0.577350269189626; }

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in) override;

    double f1d(const double &in) override;

    double f2d(const double &in) override;

    double f3d(const double &in) override;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) override;
};


#endif
