#include "ffnn/layer/OutputNNLayer.hpp"

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/unit/OutputNNUnit.hpp"


// --- Register Unit

void OutputNNLayer::_registerUnit(NetworkUnit * newUnit)
{
    NNLayer::_registerUnit(newUnit);
    if(auto * outu = dynamic_cast<OutputNNUnit *>(newUnit)) {
        _U_out.push_back(outu);
    }
}


// --- Constructor

void OutputNNLayer::construct(const int &nunits, ActivationFunctionInterface * actf)
{
    for (int i=1; i<nunits; ++i)
        {
            auto * newUnit = new OutputNNUnit(actf->getCopy());
            _registerUnit(newUnit);
        }
    delete actf;
}
