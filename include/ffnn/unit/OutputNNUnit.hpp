#ifndef OUTPUT_NN_UNIT
#define OUTPUT_NN_UNIT

#include "ffnn/unit/ShifterScalerNNUnit.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/NNRay.hpp"

// Output Neural Network Unit
class OutputNNUnit: public ShifterScalerNNUnit
{
public:
    // Constructor
    OutputNNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = NULL, const double shift = 0., const double scale = 1.) : ShifterScalerNNUnit(actf, ray, shift, scale) {}
    OutputNNUnit(const std::string &actf_id, NNRay * ray = NULL) : OutputNNUnit(std_actf::provideActivationFunction(actf_id), ray) {}
    virtual ~OutputNNUnit(){}

    // string code methods
    virtual std::string getIdCode(){return "OUT";} // return identifier for unit type

    // set output data boundaries and shift/scale accordingly
    void setOutputBounds(const double &lbound, const double &ubound);
};

#endif
