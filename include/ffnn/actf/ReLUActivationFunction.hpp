#ifndef FFNN_ACTF_RELUACTIVATIONFUNCTION_HPP
#define FFNN_ACTF_RELUACTIVATIONFUNCTION_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>


class ReLUActivationFunction: public ActivationFunctionInterface
{
protected:
    double _alpha{};

public:
    explicit ReLUActivationFunction(const double alpha = 0.){_alpha = alpha;}
    explicit ReLUActivationFunction(const std::string &params){this->setParams(params);}
    explicit ReLUActivationFunction(ReLUActivationFunction * const selu_actf) {_alpha = selu_actf->getAlpha();}

    // get copy
    ActivationFunctionInterface * getCopy() override{return new ReLUActivationFunction(_alpha);}

    // param get/set
    double getAlpha(){return _alpha;}
    void setAlpha(const double alpha){_alpha = alpha;}

    // string methods
    std::string getIdCode() override{return "RELU";}
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu() override{return 0.;}
    double getIdealInputSigma() override{return 0.577350269189626;}

    // we can use default implementation for output mu / sigma

    // computation
    double f(const double &in) override;

    double f1d(const double &in) override;

    double f2d(const double &in) override;

    double f3d(const double &in) override;

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, bool flag_d1 = false, bool flag_d2 = false, bool flag_d3 = false) override;
};


#endif
