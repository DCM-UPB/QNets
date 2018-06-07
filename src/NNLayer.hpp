#ifndef NN_LAYER
#define NN_LAYER

#include "FedNetworkLayer.hpp"
#include "NetworkLayer.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnitRay.hpp"

#include <vector>


class NNLayer: public FedNetworkLayer
{
protected:
    std::vector<NNUnit *> _U_nn; // stores pointers to all neural units

public:

    // --- Constructor

    NNLayer(const int &nunits, ActivationFunctionInterface * actf);
    void construct(const int &nunits);
    void construct(const int &nunits, ActivationFunctionInterface * actf);

    // --- Deconstructor

    ~NNLayer(){_U_nn.clear();}
    void deconstruct()
    {
        FedNetworkLayer::deconstruct();
        _U_nn.clear();
    }

    // --- Getters

    int getNNNUnits() {return _U_nn.size();}
    NNUnit * getNNUnit(const int &i) {return _U_nn[i];}

    // --- Modify structure

    void setActivationFunction(ActivationFunctionInterface * actf);

    // --- Connection

    NetworkUnitFeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) {return new NetworkUnitRay(nl);}
};


#endif
