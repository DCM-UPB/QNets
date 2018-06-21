#ifndef OUTPUT_NN_UNIT
#define OUTPUT_NN_UNIT

#include "ShifterScalerNNUnit.hpp"
#include "ActivationFunctionManager.hpp"


// Output Neural Network Unit
class OutputNNUnit: public ShifterScalerNNUnit
{
public:
    // Constructor
    OutputNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NetworkUnitFeederInterface * feeder = NULL, const double shift = 0., const double scale = 1.) : ShifterScalerNNUnit(actf, feeder, shift, scale) {}

    // string code methods
    virtual std::string getIdCode(){return "OUT";} // return identifier for unit type
};

#endif
