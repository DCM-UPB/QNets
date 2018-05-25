#include "ActivationFunctionManager.hpp"



namespace std_actf{

    std::vector<ActivationFunctionInterface *> supported_actf = {
        new IdentityActivationFunction(),
        new LogisticActivationFunction(),
        new GaussianActivationFunction(),
        new ReLUActivationFunction(),
        new SELUActivationFunction(),
        new TanSigmoidActivationFunction(),
        new SineActivationFunction()
    };

    ActivationFunctionInterface * provideActivationFunction(const std::string idcode){

        for (ActivationFunctionInterface * actf : supported_actf){
            if (idcode == actf->getIdCode()){
                return actf;
            }
        }

        return 0;
    }

}
