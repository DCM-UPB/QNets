#ifndef RELU_ACTIVATION_FUNCTION
#define RELU_ACTIVATION_FUNCTION

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>


class ReLUActivationFunction: public ActivationFunctionInterface
{
protected:
    double _alpha;

public:
    explicit ReLUActivationFunction(const double alpha = 0.){_alpha = alpha;}
    explicit ReLUActivationFunction(const std::string &params){this->setParams(params);}
    explicit ReLUActivationFunction(ReLUActivationFunction * const selu_actf) {_alpha = selu_actf->getAlpha();}

    // get copy
    ActivationFunctionInterface * getCopy(){return new ReLUActivationFunction(_alpha);}

    // param get/set
    double getAlpha(){return _alpha;}
    void setAlpha(const double alpha){_alpha = alpha;}

    // string methods
    std::string getIdCode(){return "RELU";}
    std::string getParams();
    void setParams(const std::string &params);

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 0.577350269189626;}

    // we can use default implementation for output mu / sigma

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
