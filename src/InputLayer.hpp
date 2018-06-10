#ifndef INPUT_LAYER
#define INPUT_LAYER

#include "NetworkLayer.hpp"
#include "InputUnit.hpp"

#include <vector>


class InputLayer: public NetworkLayer
{
protected:
    std::vector<InputUnit *> _U_in; // stores pointers to all input units

public:

    // --- Constructor / Destructor

    InputLayer(const int &nunits);
    void construct(const int &nunits);

    // --- Destructor

    ~InputLayer(){_U_in.clear();}
    void deconstruct(){NetworkLayer::deconstruct(); _U_in.clear();}

    // --- Getters

    int getNInputUnits() {return _U_in.size();}
    InputUnit * getInputUnit(const int &i) {return _U_in[i];}
};


#endif
