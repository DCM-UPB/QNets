#ifndef INPUT_LAYER
#define INPUT_LAYER

#include "NetworkLayer.hpp"
#include "InputUnit.hpp"

#include <vector>
#include <string>

class InputLayer: public NetworkLayer
{
protected:
    std::vector<InputUnit *> _U_in; // stores pointers to all input units

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from InputUnit and register

public:

    // --- Constructor / Destructor

    InputLayer(const int &nunits = 1){if (nunits > 1) construct(nunits);};
    void construct(const int &nunits);

    // --- Destructor

    ~InputLayer(){_U_in.clear();}
    void deconstruct(){NetworkLayer::deconstruct(); _U_in.clear();}

    // --- String Codes

    virtual std::string getIdCode(){return "INL";}

    // --- Getters

    int getNInputUnits() {return _U_in.size();}
    InputUnit * getInputUnit(const int &i) {return _U_in[i];}
};


#endif
