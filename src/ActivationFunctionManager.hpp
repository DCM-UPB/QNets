#ifndef ACTIVATION_FUNCTION_MANAGER
#define ACTIVATION_FUNCTION_MANAGER


#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"
#include "ReLUActivationFunction.hpp"
#include "SELUActivationFunction.hpp"
#include "TanSigmoidActivationFunction.hpp"
#include "SineActivationFunction.hpp"

#include <string>
#include <vector>


namespace std_actf{

    extern IdentityActivationFunction id_actf;
    extern LogisticActivationFunction lgs_actf;
    extern GaussianActivationFunction gss_actf;
    extern ReLUActivationFunction relu_actf;
    extern SELUActivationFunction selu_actf;
    extern TanSigmoidActivationFunction tans_actf;
    extern SineActivationFunction sin_actf;

    extern std::vector<ActivationFunctionInterface *> supported_actf;

    ActivationFunctionInterface * provideActivationFunction(const std::string &idCode = "lgs", const std::string &params = ""); // currently defaults to logistic actf
}



#endif
