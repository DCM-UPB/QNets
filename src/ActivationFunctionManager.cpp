#include "ActivationFunctionManager.hpp"


namespace std_actf{


    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();
    ReLUActivationFunction relu_actf = ReLUActivationFunction();
    SELUActivationFunction selu_actf = SELUActivationFunction();
    TanSigmoidActivationFunction tans_actf = TanSigmoidActivationFunction();
    SineActivationFunction sin_actf = SineActivationFunction();

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

        if (idcode == relu_actf.getIdCode()){
            return &relu_actf;
        }

        if (idcode == selu_actf.getIdCode()){
            return &selu_actf;
        }

        if (idcode == tans_actf.getIdCode()){
          return &tans_actf;
        }

        if (idcode == sin_actf.getIdCode()){
          return &sin_actf;
        }

        return 0;
    }

}
