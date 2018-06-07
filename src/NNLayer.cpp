#include "NNLayer.hpp"

#include "ActivationFunctionManager.hpp"

// --- Constructor

NNLayer::NNLayer(const int &nunits, ActivationFunctionInterface * actf)
{
    for (int i=1; i<nunits; ++i)
        {
            NNUnit * newUnit = new NNUnit(actf);
            _U.push_back(newUnit);
            _U_fed.push_back(newUnit);
            _U_nn.push_back(newUnit);
        }
}

// --- Modify structure

void NNLayer::setActivationFunction(ActivationFunctionInterface * actf)
{
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_nn.size(); ++i)
        {
            _U_nn[i]->setActivationFunction(actf);
        }
}


void NNLayer::setSize(const int &nunits)
{
    ActivationFunctionInterface * actf = _U_nn[0]->getActivationFunction();
    for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
    _U_fed.clear();
    _U_nn.clear();

    _U.push_back(_U_off);
    for (int i=1; i<nunits; ++i)
        {
            NNUnit * newUnit = new NNUnit(actf);
            _U.push_back(newUnit);
            _U_fed.push_back(newUnit);
            _U_nn.push_back(newUnit);
        }
}
