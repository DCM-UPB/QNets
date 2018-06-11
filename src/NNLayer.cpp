#include "NNLayer.hpp"

#include "ActivationFunctionInterface.hpp"
#include "NNUnit.hpp"

// --- Constructor

NNLayer::NNLayer(const int &nunits, ActivationFunctionInterface * actf)
{
    this->construct(nunits, actf);
}

void NNLayer::construct(const int &nunits)
{
    ActivationFunctionInterface * actf = _U_nn[0]->getActivationFunction();
    this->construct(nunits, actf);
}

void NNLayer::construct(const int &nunits, ActivationFunctionInterface * actf)
{
    for (int i=1; i<nunits; ++i)
        {
            NNUnit * newUnit = new NNUnit(actf->getCopy());
            _U.push_back(newUnit);
            _U_fed.push_back(newUnit);
            _U_nn.push_back(newUnit);
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
