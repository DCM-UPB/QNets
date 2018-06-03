#ifndef SHIFTER_SCALER_NN_UNIT
#define SHIFTER_SCALER_NN_UNIT

#include "ShifterScalerUnit.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "NNUnitFeederInterface.hpp"


// ShiftScaled Neural Network Unit
class ShifterScalerNNUnit: public NNUnit, public ShifterScalerUnit
{
protected:

public:

    // Constructor
    ShifterScalerNNUnit(ActivationFunctionInterface * actf, NNUnitFeederInterface * feeder = NULL, const double shift = 0., const double scale = 1.) : NNUnit(actf, feeder), ShifterScalerUnit(shift, scale) {};

    // Computation
    void computeValues() {
        NNUnit::computeValues();
        ShifterScalerUnit::_applyShiftScale();
    };
};


#endif
