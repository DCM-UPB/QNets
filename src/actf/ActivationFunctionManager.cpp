#include "ActivationFunctionManager.hpp"

#include <cstddef>

namespace std_actf{

    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();
    TanSigmoidActivationFunction tans_actf = TanSigmoidActivationFunction();
    ReLUActivationFunction relu_actf = ReLUActivationFunction();
    SELUActivationFunction selu_actf = SELUActivationFunction();
    SRLUActivationFunction srlu_actf = SRLUActivationFunction();
    SineActivationFunction sin_actf = SineActivationFunction();

    std::vector<ActivationFunctionInterface *> supported_actf = {
        &id_actf,
        &lgs_actf,
        &gss_actf,
        &tans_actf,
        &relu_actf,
        &selu_actf,
        &srlu_actf,
        &sin_actf
    };

    ActivationFunctionInterface * provideActivationFunction(const std::string &idCode, const std::string &params){
        for (ActivationFunctionInterface * actf : supported_actf){
            if (idCode == actf->getIdCode()){
                ActivationFunctionInterface * new_actf = actf->getCopy();
                new_actf->setParams(params);
                return new_actf;
            }
        }

        return NULL;
    }

}
