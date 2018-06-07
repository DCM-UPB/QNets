#include "FedNetworkLayer.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <vector>

// --- Variational Parameters

bool FedNetworkLayer::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<FedNetworkUnit *>::size_type i=0;
    NetworkUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            feeder = _U_fed[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->setVariationalParameterValue(id,vp);
                }
            i++;
        }
    return flag;
}


bool FedNetworkLayer::getVariationalParameter(const int &id, double &vp)
{
    std::vector<FedNetworkUnit *>::size_type i=0;
    NetworkUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            feeder = _U_fed[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->getVariationalParameterValue(id, vp);
                }
            i++;
        }
    return flag;
}


int FedNetworkLayer::getNVariationalParameters()
{
    int nvp=0;
    NetworkUnitFeederInterface * feeder;
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            feeder = _U_fed[i]->getFeeder();
            if (feeder)
                {
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
            id = _U_fed[i]->getFeeder()->setVariationalParametersIndexes(id);
        }
    return id;
}


// --- Connection

void FedNetworkLayer::connectOnTopOfLayer(NetworkLayer * nl)
{
    NetworkUnitFeederInterface * ray;
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            ray = this->connectUnitOnTopOfLayer(nl, i);
            if (ray) _U_fed[i]->setFeeder(ray);
        }
}

void FedNetworkLayer::disconnect()
{
    NetworkUnitFeederInterface * ray;
    for (std::vector<FedNetworkUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            ray = _U_fed[i]->getFeeder();
            delete ray;
            _U_fed[i]->setFeeder(NULL);
        }
}
