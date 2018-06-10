#ifndef OUTPUT_NN_LAYER
#define OUTPUT_NN_LAYER

#include "NNLayer.hpp"
#include "OutputNNUnit.hpp"
#include "ActivationFunctionInterface.hpp"

#include <vector>
#include <string>

class OutputNNLayer: public NNLayer
{
protected:
    std::vector<OutputNNUnit *> _U_out; // stores pointers to all output units

public:

    // --- Constructor

    OutputNNLayer(const int &nunits, ActivationFunctionInterface * actf): NNLayer(nunits, actf) {}
    void construct(const int &nunits, ActivationFunctionInterface * actf);

    // --- Destructor

    ~OutputNNLayer(){_U_out.clear();}
    void deconstruct(){NNLayer::deconstruct(); _U_out.clear();}

    // --- String Codes

    virtual std::string getIdCode(){return "outl";}

    // --- Getters

    int getNOutputNNUnits() {return _U_out.size();}
    OutputNNUnit * getOutputNNUnit(const int &i) {return _U_out[i];}
};


#endif
