#ifndef NN_LAYER
#define NN_LAYER

#include "NetworkLayerInterface.hpp"
#include "NetworkUnitRay.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"

#include <vector>


class NNLayer: public NetworkLayerInterface<NNUnit>
{
protected:

public:
    NNLayer(const int &nunits, ActivationFunctionInterface * actf);

    // --- Modify structure
    void setSize(const int &nunits);
    void setActivationFunction(ActivationFunctionInterface * actf);

};


#endif
