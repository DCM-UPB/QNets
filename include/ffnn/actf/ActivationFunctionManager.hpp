#ifndef ACTIVATION_FUNCTION_MANAGER
#define ACTIVATION_FUNCTION_MANAGER


#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/IdentityActivationFunction.hpp"
#include "ffnn/actf/LogisticActivationFunction.hpp"
#include "ffnn/actf/GaussianActivationFunction.hpp"
#include "ffnn/actf/TanSigmoidActivationFunction.hpp"
#include "ffnn/actf/ReLUActivationFunction.hpp"
#include "ffnn/actf/SELUActivationFunction.hpp"
#include "ffnn/actf/SRLUActivationFunction.hpp"
#include "ffnn/actf/SineActivationFunction.hpp"
#include "ffnn/actf/ExponentialActivationFunction.hpp"

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
