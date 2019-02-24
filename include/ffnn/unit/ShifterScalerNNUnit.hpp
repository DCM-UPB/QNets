#ifndef SHIFTER_SCALER_NN_UNIT
#define SHIFTER_SCALER_NN_UNIT

#include "ffnn/unit/ShifterScalerUnit.hpp"
#include "ffnn/unit/NNUnit.hpp"
#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/NNRay.hpp"

#include <cstddef> // for NULL
#include <string>

// ShiftScaled NNUnit
class ShifterScalerNNUnit: public NNUnit, public ShifterScalerUnit
{
public:
    // Constructor
    ShifterScalerNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = NULL, const double shift = 0., const double scale = 1.)
        : NNUnit(actf, ray), ShifterScalerUnit(shift, scale) {};
    virtual ~ShifterScalerNNUnit(){};

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (we copy NNUnit's IdealProto methods)
    virtual double getIdealProtoMu(){return NNUnit::getIdealProtoMu();}
    virtual double getIdealProtoSigma(){return NNUnit::getIdealProtoSigma();}

    // return the output mean value (mu) and standard deviation (sigma)
    virtual double getOutputMu(){return (NNUnit::getOutputMu() + _shift) * _scale;}
    virtual double getOutputSigma(){return NNUnit::getOutputSigma() * _scale;}

    // string code methods
    virtual std::string getParams(){return composeCodes(NNUnit::getParams(), ShifterScalerUnit::getParams());} // return parameter string
    virtual void setParams(const std::string &params){NNUnit::setParams(params); ShifterScalerUnit::setParams(params);}
    virtual std::string getIdCode() = 0; // this class is meant to be abstract
};

#endif
