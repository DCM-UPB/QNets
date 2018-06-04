#include "NNLayer.hpp"

#include "NetworkUnitRay.hpp"
#include "ActivationFunctionManager.hpp"

#include <iostream>


// --- Connection

void NNLayer::connectOnTopOfLayer(NetworkLayerInterface * nl)
{
    NetworkUnitFeederInterface * ray;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            ray = new NetworkUnitRay(nl);
            _U[i]->setFeeder(ray);
        }
}

void NNLayer::disconnect()
{
    NetworkUnitFeederInterface * ray;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            ray = _U[i]->getFeeder();
            delete ray;
            _U[i]->setFeeder(NULL);
        }
}


// --- Modify structure

void NNLayer::setActivationFunction(ActivationFunctionInterface * actf)
{
    _U[0]->setActivationFunction(std_actf::provideActivationFunction("lgs"));
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            _U[i]->setActivationFunction(actf);
        }
}


void NNLayer::setSize(const int &nunits)
{
    ActivationFunctionInterface * actf = _U[1]->getActivationFunction();
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
    _U.push_back(new NNUnit(std_actf::provideActivationFunction("lgs")));
    _U[0]->setProtoValue(1.);
    for (int i=1; i<nunits; ++i)
        {
            _U.push_back(new NNUnit(actf));
        }
}

// --- Constructor

NNLayer::NNLayer(const int &nunits, ActivationFunctionInterface * actf)
{
    _U.push_back(new NNUnit(std_actf::provideActivationFunction("id_")));
    _U[0]->setProtoValue(1.);

    for (int i=1; i<nunits; ++i)
        {
            _U.push_back(new NNUnit(actf));
        }
}

// --- Destructor

NNLayer::~NNLayer()
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    // clear is done by base destructor
}
