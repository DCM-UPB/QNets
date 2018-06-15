#ifndef NN_LAYER
#define NN_LAYER

#include "FedNetworkLayer.hpp"
#include "NetworkLayer.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnitRay.hpp"

#include <vector>
#include <string>

class NNLayer: public FedNetworkLayer
{
protected:
    std::vector<NNUnit *> _U_nn; // stores pointers to all neural units

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from NNUnit and register

public:

    // --- Constructor

    NNLayer(const int &nunits = 1, ActivationFunctionInterface * actf = std_actf::provideActivationFunction()){if (nunits>1) construct(nunits, actf);}
    virtual void construct(const int &nunits);
    virtual void construct(const int &nunits, ActivationFunctionInterface * actf);

    // --- Deconstructor

    virtual ~NNLayer(){_U_nn.clear();}
    virtual void deconstruct(){FedNetworkLayer::deconstruct(); _U_nn.clear();}

    // --- String Codes

    virtual std::string getIdCode(){return "NNL";}

    // --- Getters

    int getNNNUnits() {return _U_nn.size();}
    NNUnit * getNNUnit(const int &i) {return _U_nn[i];}

    // --- Modify structure

    void setActivationFunction(ActivationFunctionInterface * actf);

    // --- Connection

    virtual NetworkUnitFeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) {return new NetworkUnitRay(nl);}
};


#endif
