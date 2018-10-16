#ifndef ACTIVATION_FUNCTION_MANAGER
#define ACTIVATION_FUNCTION_MANAGER


#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"
#include "TanSigmoidActivationFunction.hpp"
#include "ReLUActivationFunction.hpp"
#include "SELUActivationFunction.hpp"
#include "SRLUActivationFunction.hpp"
#include "SineActivationFunction.hpp"
#include "ExponentialActivationFunction.hpp"

#include <string>
#include <vector>


namespace std_actf{

    extern IdentityActivationFunction id_actf;
    extern LogisticActivationFunction lgs_actf;
    extern GaussianActivationFunction gss_actf;
    extern TanSigmoidActivationFunction tans_actf;
    extern ReLUActivationFunction relu_actf;
    extern SELUActivationFunction selu_actf;
    extern SRLUActivationFunction srlu_actf;
    extern SineActivationFunction sin_actf;
    extern ExponentialActivationFunction exp_actf;

    extern std::vector<ActivationFunctionInterface *> supported_actf;

    ActivationFunctionInterface * provideActivationFunction(const std::string &idCode = "LGS", const std::string &params = ""); // currently defaults to logistic actf
}



#endif
