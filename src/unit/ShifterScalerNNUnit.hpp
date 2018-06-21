#ifndef SHIFTER_SCALER_NN_UNIT
#define SHIFTER_SCALER_NN_UNIT

#include "ShifterScalerNetworkUnit.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <cstddef> // for NULL

// ShiftScaled Neural Network Unit
class ShifterScalerNNUnit: public NNUnit, public ShifterScalerNetworkUnit
{
public:
    // Constructor
    ShifterScalerNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NetworkUnitFeederInterface * feeder = NULL, const double shift = 0., const double scale = 1.) : NNUnit(actf, feeder), ShifterScalerNetworkUnit(shift, scale) {};

    // return the output mean value (mu) and standard deviation (sigma)
    virtual double getOutputMu(){return (NNUnit::getOutputMu() + _shift) * _scale;}
    virtual double getOutputSigma(){return NNUnit::getOutputSigma() * _scale;}

    // string code methods
    virtual std::string getIdCode() = 0; // this class is meant to be abstract
};

#endif
