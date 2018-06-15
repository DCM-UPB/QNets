#ifndef FED_NETWORK_LAYER
#define FED_NETWORK_LAYER

#include "FedNetworkUnit.hpp"
#include "NetworkUnitRay.hpp"

#include <vector>

class FedNetworkLayer: public NetworkLayer
{
protected:
    std::vector<FedNetworkUnit *> _U_fed; // stores pointers to all units with feeder

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from FedNetworkUnit and register

public:

    // --- Destructor

    virtual ~FedNetworkLayer() {_U_fed.clear();}

    virtual void deconstruct(){NetworkLayer::deconstruct(); _U_fed.clear();}

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

    virtual NetworkUnitFeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) = 0; // should create and return the feeder for the given unit
    void connectOnTopOfLayer(NetworkLayer * nl);
    void disconnect();
};


#endif
