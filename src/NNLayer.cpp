#include "NNLayer.hpp"

#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "NNUnit.hpp"


// --- Register Unit

void NNLayer::_registerUnit(NetworkUnit * newUnit)
{
    FedNetworkLayer::_registerUnit(newUnit);
    if(NNUnit * nnu = dynamic_cast<NNUnit *>(newUnit)) {
        _U_nn.push_back(nnu);
    }
}


// --- Constructor

void NNLayer::construct(const int &nunits)
{
    construct(nunits, std_actf::provideActivationFunction());
}

void NNLayer::construct(const int &nunits, ActivationFunctionInterface * actf)
{
    for (int i=1; i<nunits; ++i)
        {
            NNUnit * newUnit = new NNUnit(actf->getCopy());
            _registerUnit(newUnit);
        }
    delete actf;
}


// --- Modify structure

void NNLayer::setActivationFunction(ActivationFunctionInterface * actf)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U_nn.size(); ++i)
        {
            _U_nn[i]->setActivationFunction(actf->getCopy());
        }
    delete actf;
}
