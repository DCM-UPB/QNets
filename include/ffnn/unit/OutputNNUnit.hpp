#ifndef FFNN_UNIT_OUTPUTNNUNIT_HPP
#define FFNN_UNIT_OUTPUTNNUNIT_HPP

#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/NNRay.hpp"
#include "ffnn/unit/ShifterScalerNNUnit.hpp"

// Output Neural Network Unit
class OutputNNUnit: public ShifterScalerNNUnit
{
public:
    // Constructor
    explicit OutputNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = nullptr, const double shift = 0., const double scale = 1.) : ShifterScalerNNUnit(actf, ray, shift, scale) {}
    explicit OutputNNUnit(const std::string &actf_id, NNRay * ray = nullptr) : OutputNNUnit(std_actf::provideActivationFunction(actf_id), ray) {}
    ~OutputNNUnit() override= default;

    // string code methods
    std::string getIdCode() override{return "OUT";} // return identifier for unit type

    // set output data boundaries and shift/scale accordingly
    void setOutputBounds(const double &lbound, const double &ubound);
};

#endif
