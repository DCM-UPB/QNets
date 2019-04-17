#ifndef FFNN_ACTF_TANSIGMOIDACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_TANSIGMOIDACTIVATIONFUNCTION_HPP

#include "qnets/actf/ActivationFunctionInterface.hpp"
#include <string>


class TanSigmoidActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() final { return new TanSigmoidActivationFunction(); }
    std::string getIdCode() final { return "TANS"; }

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() final { return 0.; }
    double getIdealInputSigma() final { return 0.577350269189626; }

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in) final;

    double f1d(const double &in) final;

    double f2d(const double &in) final;

    double f3d(const double &in) final;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) final;
};


#endif
