#ifndef NN_LAYER
#define NN_LAYER

#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"

#include <vector>


class NNLayer
{
protected:
    std::vector<NNUnit *> _U;

public:
    NNLayer(const int &nunits, ActivationFunctionInterface * actf);
    ~NNLayer();

    // --- Getters
    int getNUnits(){return _U.size();}
    NNUnit * getUnit(const int & i){return _U[i];}
    ActivationFunctionInterface * getActivationFunction(){return _U[1]->getActivationFunction();}

    // --- Modify structure
    void setSize(const int &nunits);
    void setActivationFunction(ActivationFunctionInterface * actf);

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
    void connectOnTopOfLayer(NNLayer * nnl);
    void disconnect();

    // --- Computation
    void computeValues();

    // --- Variational Parameters
    int getNVariationalParameters();
    bool getVariationalParameter(const int &id, double &vp);
    bool setVariationalParameter(const int &id, const double &vp);

};


#endif
