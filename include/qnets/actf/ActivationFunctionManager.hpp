#ifndef FFNN_ACTF_ACTIVATIONFUNCTIONMANAGER_HPP
#define FFNN_ACTF_ACTIVATIONFUNCTIONMANAGER_HPP


#include "qnets/actf/ActivationFunctionInterface.hpp"
#include "qnets/actf/ExponentialActivationFunction.hpp"
#include "qnets/actf/GaussianActivationFunction.hpp"
#include "qnets/actf/IdentityActivationFunction.hpp"
#include "qnets/actf/LogisticActivationFunction.hpp"
#include "qnets/actf/ReLUActivationFunction.hpp"
#include "qnets/actf/SELUActivationFunction.hpp"
#include "qnets/actf/SRLUActivationFunction.hpp"
#include "qnets/actf/SineActivationFunction.hpp"
#include "qnets/actf/TanSigmoidActivationFunction.hpp"

#include <string>
#include <vector>


namespace std_actf
{

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
} // namespace std_actf



#endif
