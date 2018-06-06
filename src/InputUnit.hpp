#ifndef INPUT_UNIT
#define INPUT_UNIT

#include "ShifterScalerNetworkUnit.hpp"

// Input Unit
class InputUnit: public ShifterScalerNetworkUnit
{
protected:
    const int _index;

public:
    InputUnit(const int index): _index(index) {} // the index of the input unit, i.e. d/dx_index f(_pv) = 1

    // Computation
    void computeFeed(){}
    void computeActivation(){}
    void computeDerivatives(){}

    void computeValues() {_v = _pv; if (_v1d) _v1d[_index] = 1.;}
};


#endif
