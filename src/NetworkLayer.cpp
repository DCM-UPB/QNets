#include "NetworkLayer.hpp"
#include "OffsetUnit.hpp"
#include "NetworkUnit.hpp"

#include <vector>

// --- Constructor

NetworkLayer::NetworkLayer()
{
    _U_off = new OffsetUnit();
    _U.push_back(_U_off);
}


// --- Destructor

NetworkLayer::~NetworkLayer()
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        delete _U[i];
    }
    _U.clear();
}

void NetworkLayer::deconstruct()
{
    for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
    _U.push_back(_U_off);
}


// --- Modify structure

void NetworkLayer::setSize(const int &nunits)
{
    this->deconstruct();
    this->construct(nunits);
}


// --- Values to compute


void NetworkLayer::addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        _U[i]->setCrossSecondDerivativeSubstrate(nx0, nvp);
    }
}


void NetworkLayer::addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        _U[i]->setCrossFirstDerivativeSubstrate(nx0, nvp);
    }
}


void NetworkLayer::addVariationalFirstDerivativeSubstrate(const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        _U[i]->setVariationalFirstDerivativeSubstrate(nvp);
    }
}


void NetworkLayer::addSecondDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        _U[i]->setSecondDerivativeSubstrate(nx0);
    }
}


void NetworkLayer::addFirstDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
        _U[i]->setFirstDerivativeSubstrate(nx0);
    }
}


// --- Computation

void NetworkLayer::computeValues()
{
#ifdef OPENMP
#pragma omp for schedule(static, 1)
#endif
    for (NetworkUnit * u : _U) u->computeValues();
}
