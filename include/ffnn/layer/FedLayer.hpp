#ifndef FED_NETWORK_LAYER
#define FED_NETWORK_LAYER

#include "FedUnit.hpp"
#include "NNRay.hpp"

#include <vector>

class FedLayer: public NetworkLayer
{
protected:
    std::vector<FedUnit *> _U_fed; // stores pointers to all units with feeder

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from FedUnit and register

public:
    // --- Destructor

    virtual ~FedLayer() {_U_fed.clear();}

    virtual void deconstruct(){NetworkLayer::deconstruct(); _U_fed.clear();}

    // --- Getters

    int getNFedUnits() {return _U_fed.size();}
    FedUnit * getFedUnit(const int &i) {return _U_fed[i];}


    // --- Variational Parameters

    bool setVariationalParameter(const int &id, const double &vp);
    bool getVariationalParameter(const int &id, double &vp);
    int getNVariationalParameters();
    int getMaxVariationalParameterIndex();

    // --- Values to compute

    int setVariationalParametersID(const int &id_vp);

    // --- Connection

    virtual FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) = 0; // should create and return the feeder for the given unit
    void connectOnTopOfLayer(NetworkLayer * nl);
    void disconnect();
};

#endif
