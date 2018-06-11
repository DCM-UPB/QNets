#ifndef NN_UNIT
#define NN_UNIT

#include "FedNetworkUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "NetworkUnitFeederInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include <cstddef>

// Unit of an Artificial Neural Network
class NNUnit: public FedNetworkUnit
{
protected:

    // Activation Function of the unit
    // A function that calculates the output value from the input value (protovalue)
    ActivationFunctionInterface * _actf; // activation function

public:
    // Constructor and destructor
    NNUnit(ActivationFunctionInterface * actf, NetworkUnitFeederInterface * feeder = NULL) : FedNetworkUnit(feeder) {_actf = actf;}
    virtual ~NNUnit(){ if (_actf) delete _actf; _actf = NULL; }

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf){if (_actf) delete _actf; _actf=actf;}

    // Getters
    ActivationFunctionInterface * getActivationFunction(){return _actf;}

    // Computation
    void computeOutput();
};


#endif
