#ifndef SELU_ACTIVATION_FUNCTION
#define SELU_ACTIVATION_FUNCTION

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include <string>

class SELUActivationFunction: public ActivationFunctionInterface
{
protected:
    double _alpha, _lambda;

public:
    SELUActivationFunction(const double alpha = 1.6733, const double lambda = 1.0507){_alpha = alpha; _lambda = lambda;}
    explicit SELUActivationFunction(const std::string &params){this->setParams(params);}
    explicit SELUActivationFunction(SELUActivationFunction * const selu_actf) {_alpha = selu_actf->getAlpha(); _lambda = selu_actf->getLambda();}

    // get copy
    ActivationFunctionInterface * getCopy(){return new SELUActivationFunction(_alpha, _lambda);}

    // param get/set
    double getAlpha(){return _alpha;}
    double getLambda(){return _lambda;}
    void setAlpha(const double alpha){_alpha = alpha;}
    void setLambda(const double lambda){_lambda = lambda;}

    // string methods
    std::string getIdCode(){return "SELU";}
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
