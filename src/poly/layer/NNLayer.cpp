#include "qnets/poly/layer/NNLayer.hpp"


// --- Register Unit

void NNLayer::_registerUnit(NetworkUnit * newUnit)
{
    FedLayer::_registerUnit(newUnit);
    if (auto * nnu = dynamic_cast<NNUnit *>(newUnit)) {
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
    for (int i = 1; i < nunits; ++i) {
        auto * newUnit = new NNUnit(actf->getCopy());
        _registerUnit(newUnit);
    }
    delete actf;
}


// --- Modify structure

void NNLayer::setActivationFunction(ActivationFunctionInterface * actf)
{
    for (auto &i : _U_nn) {
        i->setActivationFunction(actf->getCopy());
    }
    delete actf;
}
