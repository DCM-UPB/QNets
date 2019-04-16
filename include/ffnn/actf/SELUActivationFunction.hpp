#ifndef FFNN_ACTF_SELUACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_SELUACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>

class SELUActivationFunction: public ActivationFunctionInterface
{
protected:
    double _alpha{}, _lambda{};

public:
    explicit SELUActivationFunction(const double alpha = 1.6733, const double lambda = 1.0507)
    {
        _alpha = alpha;
        _lambda = lambda;
    }
    explicit SELUActivationFunction(const std::string &params) { this->setParams(params); }
    explicit SELUActivationFunction(SELUActivationFunction * const selu_actf)
    {
        _alpha = selu_actf->getAlpha();
        _lambda = selu_actf->getLambda();
    }

    // get copy
    ActivationFunctionInterface * getCopy() final { return new SELUActivationFunction(_alpha, _lambda); }

    // param get/set
    double getAlpha() { return _alpha; }
    double getLambda() { return _lambda; }
    void setAlpha(const double alpha) { _alpha = alpha; }
    void setLambda(const double lambda) { _lambda = lambda; }

    // string methods
    std::string getIdCode() final { return "SELU"; }
    std::string getParams() final;
    void setParams(const std::string &params) final;

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() final { return 0.; }
    double getIdealInputSigma() final { return 0.577350269189626; }

    // we can use default implementation for output mu / sigma

    // computation
    double f(const double &in) final;

    double f1d(const double &in) final;

    double f2d(const double &in) final;

    double f3d(const double &in) final;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) final;
};


#endif
