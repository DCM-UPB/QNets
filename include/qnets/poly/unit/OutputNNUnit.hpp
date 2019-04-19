#ifndef FFNN_UNIT_OUTPUTNNUNIT_HPP
#define FFNN_UNIT_OUTPUTNNUNIT_HPP

#include "qnets/poly/actf/ActivationFunctionManager.hpp"
#include "qnets/poly/feed/NNRay.hpp"
#include "qnets/poly/unit/ShifterScalerNNUnit.hpp"

// Output Neural Network Unit
class OutputNNUnit: public ShifterScalerNNUnit
{
public:
    // Constructor
    explicit OutputNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = nullptr, const double shift = 0., const double scale = 1.):
            ShifterScalerNNUnit(actf, ray, shift, scale) {}
    explicit OutputNNUnit(const std::string &actf_id, NNRay * ray = nullptr):
            OutputNNUnit(std_actf::provideActivationFunction(actf_id), ray) {}
    ~OutputNNUnit() final = default;

    // string code methods
    std::string getIdCode() final { return "OUT"; } // return identifier for unit type

    // set output data boundaries and shift/scale accordingly
    void setOutputBounds(const double &lbound, const double &ubound);
};

#endif
