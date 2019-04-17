#ifndef FFNN_LAYER_FEDLAYER_HPP
#define FFNN_LAYER_FEDLAYER_HPP

#include "qnets/feed/NNRay.hpp"
#include "qnets/unit/FedUnit.hpp"

#include <vector>

class FedLayer: public NetworkLayer
{
protected:
    std::vector<FedUnit *> _U_fed; // stores pointers to all units with feeder

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from FedUnit and register

public:
    // --- Destructor

    ~FedLayer() override { _U_fed.clear(); }

    void deconstruct() override
    {
        NetworkLayer::deconstruct();
        _U_fed.clear();
    }

    // --- Getters

    int getNFedUnits() { return _U_fed.size(); }
    FedUnit * getFedUnit(const int &i) { return _U_fed[i]; }


    // --- Variational Parameters

    bool setVariationalParameter(const int &id, const double &vp) override;
    bool getVariationalParameter(const int &id, double &vp) override;
    int getNVariationalParameters() override;
    int getMaxVariationalParameterIndex() override;

    // --- Values to compute

    int setVariationalParametersID(const int &id_vp) override;

    // --- Connection

    virtual FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) = 0; // should create and return the feeder for the given unit
    void connectOnTopOfLayer(NetworkLayer * nl);
    void disconnect();

    // --- Computation 
    void computeValues() override; // overriding to add OMP pragma
};

#endif
