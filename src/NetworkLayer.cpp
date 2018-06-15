#include "NetworkLayer.hpp"
#include "OffsetUnit.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>

// --- Constructor

NetworkLayer::NetworkLayer()
{
    _U_off = new OffsetUnit();
    _registerUnit(_U_off);
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
    _registerUnit(_U_off);
}


// --- String Code methods

std::string NetworkLayer::getMemberTreeCode()
{
    std::vector<std::string> unitCodes;
    for (NetworkUnit * u : _U) unitCodes.push_back(u->getTreeCode());
    return composeCodeList(unitCodes);
}


void NetworkLayer::setMemberParams(const std::string &memberTreeCode)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i) {
        _U[i]->setTreeParams(readTreeCode(memberTreeCode, i));
    }
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
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i) _U[i]->computeValues();
}
