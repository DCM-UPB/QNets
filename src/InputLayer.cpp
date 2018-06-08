#include "InputLayer.hpp"
#include "InputUnit.hpp"

// --- Constructor

InputLayer::InputLayer(const int &nunits)
{
    this->construct(nunits);
}

void InputLayer::construct(const int &nunits)
{
    for (int i=1; i<nunits; ++i)
        {
            InputUnit * newUnit = new InputUnit(i-1);
            _U.push_back(newUnit);
            _U_in.push_back(newUnit);
        }
}
