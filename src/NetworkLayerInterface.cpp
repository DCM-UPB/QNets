#include "NetworkLayerInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <vector>

// --- Variational Parameters

bool NetworkLayerInterface::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<NetworkUnit *>::size_type i=1;
    NetworkUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U.size()) )
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->setVariationalParameterValue(id,vp);
                }
            i++;
        }
    return flag;
}


bool NetworkLayerInterface::getVariationalParameter(const int &id, double &vp)
{
    std::vector<NetworkUnit *>::size_type i=1;
    NetworkUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U.size()) )
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->getVariationalParameterValue(id, vp);
                }
            i++;
        }
    return flag;
}


int NetworkLayerInterface::getNVariationalParameters()
{
    int nvp=0;
    NetworkUnitFeederInterface * feeder;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    nvp += feeder->getNVariationalParameters();
                }
        }
    return nvp;
}


// --- Computation

void NetworkLayerInterface::computeValues()
{
    for (NetworkUnit * u : _U) u->computeValues();
}


// --- Values to compute

int NetworkLayerInterface::setVariationalParametersID(const int &id_vp)
{
    int id = id_vp;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            id = _U[i]->getFeeder()->setVariationalParametersIndexes(id);
        }
    return id;
}


void NetworkLayerInterface::addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setCrossSecondDerivativeSubstrate(nx0, nvp);
        }
}


void NetworkLayerInterface::addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setCrossFirstDerivativeSubstrate(nx0, nvp);
        }
}


void NetworkLayerInterface::addVariationalFirstDerivativeSubstrate(const int &nvp)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setVariationalFirstDerivativeSubstrate(nvp);
        }
}


void NetworkLayerInterface::addSecondDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setSecondDerivativeSubstrate(nx0);
        }
}


void NetworkLayerInterface::addFirstDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setFirstDerivativeSubstrate(nx0);
        }
}

// --- Destructor

NetworkLayerInterface::~NetworkLayerInterface()
{
    _U.clear();
}
