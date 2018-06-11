#include "ActivationFunctionManager.hpp"

#include <cstddef>

namespace std_actf{

    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();
    ReLUActivationFunction relu_actf = ReLUActivationFunction();
    SELUActivationFunction selu_actf = SELUActivationFunction();
    TanSigmoidActivationFunction tans_actf = TanSigmoidActivationFunction();
    SineActivationFunction sin_actf = SineActivationFunction();

    std::vector<ActivationFunctionInterface *> supported_actf = {
        &id_actf,
        &lgs_actf,
        &gss_actf,
        &relu_actf,
        &selu_actf,
        &tans_actf,
        &sin_actf
    };

    ActivationFunctionInterface * provideActivationFunction(const std::string &idcode, const std::string &params){

        for (ActivationFunctionInterface * actf : supported_actf){
            if (idcode == actf->getIdCode()){
                ActivationFunctionInterface * new_actf = actf->getCopy();
                new_actf->setParams(params);
                return new_actf;
            }
        }

        return NULL;
    }

}
