#ifndef RELU_ACTIVATION_FUNCTION
#define RELU_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class ReLUActivationFunction: public ActivationFunctionInterface
{
protected:
    double _alpha;

public:
    ReLUActivationFunction(const double alpha = 0.){_alpha = alpha;}
    ReLUActivationFunction(const std::string &params){this->setParams(params);}
    ReLUActivationFunction(ReLUActivationFunction * const selu_actf) {_alpha = selu_actf->getAlpha();}

    // get copy
    ActivationFunctionInterface * getCopy(){return new ReLUActivationFunction(_alpha);}

    // param getters
    double getAlpha(){return _alpha;}
    void setAlpha(const double alpha){_alpha = alpha;}

    // string methods
    std::string getIdCode(){return "relu";}
    std::string getParams();
    void setParams(const std::string &params);

    // computation
    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
