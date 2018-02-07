#include "ActivationFunctionManager.hpp"




namespace std_activation_function{


    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();


    ActivationFunctionInterface * provideActivationFunction(const std::string idcode){
        if (idcode == (new IdentityActivationFunction)->getIdCode()){
            return &id_actf;
        }

        if (idcode == (new LogisticActivationFunction)->getIdCode()){
            return &lgs_actf;
        }

        if (idcode == (new GaussianActivationFunction)->getIdCode()){
            return &gss_actf;
        }

        return 0;
    }

}
