#include "FedNetworkLayer.hpp"
#include "NetworkUnitFeederInterface.hpp"
#include "FedNetworkUnit.hpp"

#include <vector>


// --- Register Unit

void FedNetworkLayer::_registerUnit(NetworkUnit * newUnit)
{
    NetworkLayer::_registerUnit(newUnit);
    if(FedNetworkUnit * fu = dynamic_cast<FedNetworkUnit *>(newUnit)) {
        _U_fed.push_back(fu);
    }
}


// --- Variational Parameters

bool FedNetworkLayer::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<FedNetworkUnit *>::size_type i=0;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            NetworkUnitFeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                flag = feeder->setVariationalParameterValue(id,vp);
            }
            i++;
        }
    return flag;
}


bool FedNetworkLayer::getVariationalParameter(const int &id, double &vp)
{
    std::vector<FedNetworkUnit *>::size_type i=0;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            NetworkUnitFeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                flag = feeder->getVariationalParameterValue(id, vp);
            }
            i++;
        }
    return flag;
}


int FedNetworkLayer::getNVariationalParameters()
{
    int nvp=0;
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            NetworkUnitFeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                nvp += feeder->getNVariationalParameters();
            }
        }
    return nvp;
}


// --- Values to compute

int FedNetworkLayer::setVariationalParametersID(const int &id_vp)
{
    int id = id_vp;
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            NetworkUnitFeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                id = _U_fed[i]->getFeeder()->setVariationalParametersIndexes(id);
            }
        }
    return id;
}


// --- Connection

void FedNetworkLayer::connectOnTopOfLayer(NetworkLayer * nl)
{
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            NetworkUnitFeederInterface * ray = this->connectUnitOnTopOfLayer(nl, i);
            if (ray) _U_fed[i]->setFeeder(ray);
        }
}

void FedNetworkLayer::disconnect()
{
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            delete _U_fed[i]->getFeeder();
            _U_fed[i]->setFeeder(NULL);
        }
}
