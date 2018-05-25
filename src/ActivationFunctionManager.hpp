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

    extern std::vector<ActivationFunctionInterface *> supported_actf;

    ActivationFunctionInterface * provideActivationFunction(const std::string idcode);
}



#endif
