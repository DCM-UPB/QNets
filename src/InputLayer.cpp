#include "InputLayer.hpp"
#include "InputUnit.hpp"


// --- Register Unit

void InputLayer::_registerUnit(NetworkUnit * u)
{
    NetworkLayer::_registerUnit(u);
    if(InputUnit * inu = dynamic_cast<InputUnit *>(u)) {
        _U_in.push_back(inu);
    }
}


// --- Construct

void InputLayer::construct(const int &nunits)
{
    for (int i=1; i<nunits; ++i)
        {
            InputUnit * newUnit = new InputUnit(i-1);
            _registerUnit(newUnit);
        }
}
