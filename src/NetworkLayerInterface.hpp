#ifndef NETWORK_LAYER_INTERFACE
#define NETWORK_LAYER_INTERFACE

#include "NetworkUnit.hpp"

#include <vector>
#include <memory>


class NetworkLayerInterface
{
protected:
    std::vector<NetworkUnit *> _U;

public:
    NetworkLayerInterface() {};
    virtual ~NetworkLayerInterface();

    // --- Getters
    int getNUnits(){return _U.size();}
    NetworkUnit * getUnit(const int & i){return _U[i];}

    // --- Modify structure
    virtual void setSize(const int &nunits) = 0;

    // --- Values to compute
    void addFirstDerivativeSubstrate(const int &nx0);
    /* add the first derivative substrate to all units
       nx0 is the number of units used as input, i.e.
       the number of derivatives that will be computed */

    void addSecondDerivativeSubstrate(const int &nx0);
    /* add the second derivative substrate to all units
       nx0 is the number of units used as input, i.e.
       the number of derivatives that will be computed */

    void addVariationalFirstDerivativeSubstrate(const int &nvp);
    /* add the variational first derivative substrate to all units.
       nvp is the number of variational parameters in the NN */

    void addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp);
    /* add the cross first derivative substrate to all units.
       nx0 is the number of units used as input
       nvp is the number of variational parameters in the NN */

    void addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp);
    /* add the cross second derivative substrate to all units.
       nx0 is the number of units used as input
       nvp is the number of variational parameters in the NN */

    int setVariationalParametersID(const int &id_vp);   // assign the id of the variational parameters to the feeders

    // --- Connection
    virtual void connectOnTopOfLayer(NetworkLayerInterface * nl) = 0;
    virtual void disconnect() = 0;

    // --- Computation
    void computeValues();

    // --- Variational Parameters
    int getNVariationalParameters();
    bool getVariationalParameter(const int &id, double &vp);
    bool setVariationalParameter(const int &id, const double &vp);

};


#endif
