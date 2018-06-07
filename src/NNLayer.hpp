#ifndef NN_LAYER
#define NN_LAYER

#include "FedNetworkLayer.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"

#include <vector>


class NNLayer: public FedNetworkLayer
{
protected:
    std::vector<NNUnit *> _U_nn; // stores pointers to all neural units

public:

    // --- Constructor / Destructor

    NNLayer(const int &nunits, ActivationFunctionInterface * actf);
    ~NNLayer(){_U_nn.clear();}

    // --- Getters

    int getNNNUnits() {return _U_nn.size();}
    NNUnit * getNNUnit(const int &i) {return _U_nn[i];}

    // --- Modify structure
    void setSize(const int &nunits);
    void setActivationFunction(ActivationFunctionInterface * actf);
};


#endif
