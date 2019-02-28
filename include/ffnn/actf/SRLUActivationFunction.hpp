#ifndef FFNN_ACTF_SRLUACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_SRLUACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>

// Smooth Rectified Linear Unit ( == ln(1+exp(x)) )
class SRLUActivationFunction: public ActivationFunctionInterface
{
public:
    // getters
    ActivationFunctionInterface * getCopy() override{return new SRLUActivationFunction();}
    std::string getIdCode() override{return "SRLU";}

    // input should be in the rage [-5 : 5] -> mu=0   sigma=10/sqrt(12)
    double getIdealInputMu() override{return 0.;}
    double getIdealInputSigma() override{return 2.886751345948129;}

    // we can use default implementation for output mu/sigma

    // computation
    double f(const double &in) override;

    double f1d(const double &in) override;

    double f2d(const double &in) override;

    double f3d(const double &in) override;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) override;
};

#endif
