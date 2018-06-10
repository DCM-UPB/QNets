#ifndef OUTPUT_NN_UNIT
#define OUTPUT_NN_UNIT

#include "ShifterScalerNNUnit.hpp"

// Output Neural Network Unit
class OutputNNUnit: public ShifterScalerNNUnit
{
protected:

public:

    // Constructor
    OutputNNUnit(ActivationFunctionInterface * actf, NetworkUnitFeederInterface * feeder = NULL, const double shift = 0., const double scale = 1.) : ShifterScalerNNUnit(actf, feeder, shift, scale) {};

    // string code methods
    virtual std::string getIdCode(){return "out";} // return identifier for unit type
};

#endif
