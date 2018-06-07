#ifndef FED_NETWORK_LAYER
#define FED_NETWORK_LAYER

#include "FedNetworkUnit.hpp"
#include "NetworkUnitRay.hpp"

#include <vector>

class FedNetworkLayer: public NetworkLayerInterface
{
protected:
    std::vector<FedNetworkUnit *> _U_fed; // stores pointers to all units with feeder

public:

    // --- Destructor

    virtual ~FedNetworkLayer() {_U_fed.clear();}

    virtual void deconstruct()
    {
        NetworkLayerInterface::deconstruct();
        _U_fed.clear();
    }

    // --- Getters

    int getNFedUnits() {return _U_fed.size();}
    FedNetworkUnit * getFedUnit(const int &i) {return _U_fed[i];}


    // --- Variational Parameters

    bool setVariationalParameter(const int &id, const double &vp);
    bool getVariationalParameter(const int &id, double &vp);
    int getNVariationalParameters();

    // --- Values to compute

    int setVariationalParametersID(const int &id_vp);

    // --- Connection

    virtual NetworkUnitFeederInterface * connectUnitOnTopOfLayer(NetworkLayerInterface * nl, const int &i) = 0; // should create and return the feeder for the given unit
    void connectOnTopOfLayer(NetworkLayerInterface * nl);
    void disconnect();
};


#endif
