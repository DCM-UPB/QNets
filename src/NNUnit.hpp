#ifndef NN_UNIT
#define NN_UNIT

#include "NetworkUnitInterface.hpp"
#include "ActivationFunctionInterface.hpp"
#include "NNUnitFeederInterface.hpp"


// Unit of an Artificial Neural Network
class NNUnit: public NetworkUnitInterface
{
protected:

    // Activation Function of the unit
    // A function that calculates the output value from the input value (protovalue)
    ActivationFunctionInterface * _actf; // activation function

    // Feeder of the unit
    // The feeder of a unit is a class that takes care of providing the input to the unit, when called via: _feeder->getFeed()
    NNUnitFeederInterface * _feeder;

public:
    // Constructor and destructor
    NNUnit(ActivationFunctionInterface * actf, NNUnitFeederInterface * feeder = NULL);
    ~NNUnit();

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf){_actf=actf;}
    void setFeeder(NNUnitFeederInterface * feeder){_feeder = feeder;}

    // Getters
    ActivationFunctionInterface * getActivationFunction(){return _actf;}
    NNUnitFeederInterface * getFeeder(){return _feeder;}

    // Computation
    void computeValues();
};


#endif
