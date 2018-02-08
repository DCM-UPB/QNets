#include "ActivationFunctionManager.hpp"

#include <stdexcept>



namespace std_actf{


    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();


    ActivationFunctionInterface * provideActivationFunction(const std::string idcode){
        if (idcode == id_actf.getIdCode()){
            return &id_actf;
        }

        if (idcode == lgs_actf.getIdCode()){
            return &lgs_actf;
        }

        if (idcode == gss_actf.getIdCode()){
            return &gss_actf;
        }

        throw std::invalid_argument( "could not find an activation function with the provided idcode: " + idcode );
    }

}
