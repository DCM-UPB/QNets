#ifndef FFNN_ACTF_SRLUACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_SRLUACTIVATIONFUNCTION_HPP

#include "qnets/actf/ActivationFunctionInterface.hpp"
#include <string>

// Smooth Rectified Linear Unit ( == ln(1+exp(x)) )
class SRLUActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() final { return new SRLUActivationFunction(); }
    std::string getIdCode() final { return "SRLU"; }

    // input should be in the rage [-5 : 5] -> mu=0   sigma=10/sqrt(12)
    double getIdealInputMu() final { return 0.; }
    double getIdealInputSigma() final { return 2.886751345948129; }

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in) final;

    double f1d(const double &in) final;

    double f2d(const double &in) final;

    double f3d(const double &in) final;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) final;
};

#endif
