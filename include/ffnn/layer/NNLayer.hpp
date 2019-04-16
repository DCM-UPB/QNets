#ifndef FFNN_LAYER_NNLAYER_HPP
#define FFNN_LAYER_NNLAYER_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/feed/NNRay.hpp"
#include "ffnn/layer/FedLayer.hpp"
#include "ffnn/layer/NetworkLayer.hpp"
#include "ffnn/unit/NNUnit.hpp"

#include <string>
#include <vector>

class NNLayer: public FedLayer
{
protected:
    std::vector<NNUnit *> _U_nn; // stores pointers to all neural units

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from NNUnit and register
public:
    // --- Constructor

    explicit NNLayer(const int &nunits = 1, ActivationFunctionInterface * actf = std_actf::provideActivationFunction())
    {
        if (nunits > 1) {
            construct(nunits, actf);
        }
    }
    void construct(const int &nunits) override;
    virtual void construct(const int &nunits, ActivationFunctionInterface * actf);

    // --- Deconstructor

    ~NNLayer() override { _U_nn.clear(); }
    void deconstruct() override
    {
        FedLayer::deconstruct();
        _U_nn.clear();
    }

    // --- String Codes

    std::string getIdCode() override { return "NNL"; }

    // --- Getters

    int getNNeuralUnits() { return _U_nn.size(); }
    NNUnit * getNNUnit(const int &i) { return _U_nn[i]; }

    // --- Modify structure

    void setActivationFunction(ActivationFunctionInterface * actf);

    // --- Connection

    FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int & /*i*/) override { return new NNRay(nl); }
};


#endif
