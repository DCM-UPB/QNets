#ifndef FFNN_LAYER_OUTPUTNNLAYER_HPP
#define FFNN_LAYER_OUTPUTNNLAYER_HPP

#include "qnets/poly/actf/ActivationFunctionInterface.hpp"
#include "qnets/poly/actf/ActivationFunctionManager.hpp"
#include "qnets/poly/layer/NNLayer.hpp"
#include "qnets/poly/unit/OutputNNUnit.hpp"

#include <string>
#include <vector>

class OutputNNLayer: public NNLayer
{
protected:
    std::vector<OutputNNUnit *> _U_out; // stores pointers to all output units

    void _registerUnit(NetworkUnit * newUnit); // check if NetworkUnit is a/derived from OutputNNUnit and register

public:
    // --- Constructor

    explicit OutputNNLayer(const int &nunits = 1, ActivationFunctionInterface * actf = std_actf::provideActivationFunction()):
            NNLayer(0, actf)
    {
        if (nunits > 1) {
            construct(nunits, actf);
        }
    }
    void construct(const int &nunits, ActivationFunctionInterface * actf) override;

    // --- Destructor

    ~OutputNNLayer() override { _U_out.clear(); }
    void deconstruct() override
    {
        NNLayer::deconstruct();
        _U_out.clear();
    }

    // --- String Codes

    std::string getIdCode() override { return "OUTL"; }

    // --- Getters

    int getNOutputNNUnits() { return _U_out.size(); }
    OutputNNUnit * getOutputNNUnit(const int &i) { return _U_out[i]; }
};

#endif
