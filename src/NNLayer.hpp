#ifndef NN_LAYER
#define NN_LAYER

#include "NetworkLayerInterface.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"

#include <vector>


class NNLayer: public NetworkLayerInterface<NNUnit>
{
protected:

public:

    // --- Constructor
    //template <typename NNUnitType>
    NNLayer(const int &nunits, ActivationFunctionInterface * actf)
    {
        _U.push_back(new NNUnit(std_actf::provideActivationFunction("id_")));
        _U[0]->setProtoValue(1.);

        for (int i=1; i<nunits; ++i)
            {
                _U.push_back(new NNUnit(actf));
            }
    }


    // --- Modify structure
    void setSize(const int &nunits);
    void setActivationFunction(ActivationFunctionInterface * actf);
};


#endif
