#ifndef INPUT_LAYER
#define INPUT_LAYER

#include "NetworkLayerInterface.hpp"
#include "InputUnit.hpp"

#include <vector>


class InputLayer: public NetworkLayerInterface
{
protected:
    std::vector<InputUnit *> _U_in; // stores pointers to all input units

public:

    // --- Constructor / Destructor

    InputLayer(const int &nunits);
    ~InputLayer(){_U_in.clear();}

    // --- Getters

    int getNInputUnits() {return _U_in.size();}
    InputUnit * getInputUnit(const int &i) {return _U_in[i];}

    // --- Modify structure
    void setSize(const int &nunits);
};


#endif
