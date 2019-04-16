#ifndef FFNN_LAYER_INPUTLAYER_HPP
#define FFNN_LAYER_INPUTLAYER_HPP

#include "ffnn/layer/NetworkLayer.hpp"
#include "ffnn/unit/InputUnit.hpp"

#include <string>
#include <vector>

class InputLayer: public NetworkLayer
{
protected:
    std::vector<InputUnit *> _U_in; // stores pointers to all input units

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from InputUnit and register

public:
    // --- Constructor / Destructor

    explicit InputLayer(const int &nunits = 1)
    {
        if (nunits > 1) {
            construct(nunits);
        }
    };
    void construct(const int &nunits) override;

    // --- Destructor

    ~InputLayer() override { _U_in.clear(); }
    void deconstruct() override
    {
        NetworkLayer::deconstruct();
        _U_in.clear();
    }

    // --- String Codes

    std::string getIdCode() override { return "INL"; }

    // --- Getters

    int getNInputUnits() { return _U_in.size(); }
    InputUnit * getInputUnit(const int &i) { return _U_in[i]; }
};

#endif
