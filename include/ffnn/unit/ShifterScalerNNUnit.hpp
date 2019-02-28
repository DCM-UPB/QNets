#ifndef FFNN_UNIT_SHIFTERSCALERNNUNIT_HPP
#define FFNN_UNIT_SHIFTERSCALERNNUNIT_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/NNRay.hpp"
#include "ffnn/unit/NNUnit.hpp"
#include "ffnn/unit/ShifterScalerUnit.hpp"

#include <cstddef> // for NULL
#include <string>

// ShiftScaled NNUnit
class ShifterScalerNNUnit: public NNUnit, public ShifterScalerUnit
{
public:
    // Constructor
    explicit ShifterScalerNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = nullptr, const double shift = 0., const double scale = 1.)
        : NNUnit(actf, ray), ShifterScalerUnit(shift, scale) {};
    ~ShifterScalerNNUnit() override= default;;

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (we copy NNUnit's IdealProto methods)
    double getIdealProtoMu() override{return NNUnit::getIdealProtoMu();}
    double getIdealProtoSigma() override{return NNUnit::getIdealProtoSigma();}

    // return the output mean value (mu) and standard deviation (sigma)
    double getOutputMu() override{return (NNUnit::getOutputMu() + _shift) * _scale;}
    double getOutputSigma() override{return NNUnit::getOutputSigma() * _scale;}

    // string code methods
    std::string getParams() override{return composeCodes(NNUnit::getParams(), ShifterScalerUnit::getParams());} // return parameter string
    void setParams(const std::string &params) override{NNUnit::setParams(params); ShifterScalerUnit::setParams(params);}
    std::string getIdCode() override = 0; // this class is meant to be abstract
};

#endif
