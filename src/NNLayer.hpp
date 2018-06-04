#ifndef NN_LAYER
#define NN_LAYER

#include "NetworkLayerInterface.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"

#include <vector>


class NNLayer: public NetworkLayerInterface
{
protected:
    std::vector<NNUnit *> _U;

public:
    NNLayer(const int &nunits, ActivationFunctionInterface * actf);
    ~NNLayer();

    // --- Getters
    NNUnit * getUnit(const int & i){return _U[i];}
    ActivationFunctionInterface * getActivationFunction(){return _U[1]->getActivationFunction();} // this kind of assumes that all units have the same actf. I would rather like to get rid of it

    // --- Modify structure
    void setSize(const int &nunits);
    void setActivationFunction(ActivationFunctionInterface * actf);

    // --- Connection
    void connectOnTopOfLayer(NetworkLayerInterface * nl);
    void disconnect();
};


#endif
