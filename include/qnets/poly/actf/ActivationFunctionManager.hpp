#ifndef FFNN_ACTF_ACTIVATIONFUNCTIONMANAGER_HPP
#define FFNN_ACTF_ACTIVATIONFUNCTIONMANAGER_HPP


#include "qnets/poly/actf/ActivationFunctionInterface.hpp"
#include "qnets/poly/actf/ExponentialActivationFunction.hpp"
#include "qnets/poly/actf/GaussianActivationFunction.hpp"
#include "qnets/poly/actf/IdentityActivationFunction.hpp"
#include "qnets/poly/actf/LogisticActivationFunction.hpp"
#include "qnets/poly/actf/ReLUActivationFunction.hpp"
#include "qnets/poly/actf/SELUActivationFunction.hpp"
#include "qnets/poly/actf/SRLUActivationFunction.hpp"
#include "qnets/poly/actf/SineActivationFunction.hpp"
#include "qnets/poly/actf/TanSigmoidActivationFunction.hpp"

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
