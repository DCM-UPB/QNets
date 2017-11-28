#include "ActivationFunctionManager.hpp"

#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"



ActivationFunctionInterface * ActivationFunctionManager::provideActivationFunction(const std::string idcode){
        
    if (idcode == (new IdentityActivationFunction)->getIdCode()){
        return new IdentityActivationFunction;
    }
    
    if (idcode == (new LogisticActivationFunction)->getIdCode()){
        return new LogisticActivationFunction;
    }
    
    if (idcode == (new GaussianActivationFunction)->getIdCode()){
        return new GaussianActivationFunction;
    }
    
    return 0;
}
