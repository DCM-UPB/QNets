#include "qnets/layer/InputLayer.hpp"


// --- Register Unit

void InputLayer::_registerUnit(NetworkUnit * newUnit)
{
    NetworkLayer::_registerUnit(newUnit);
    if (auto * inu = dynamic_cast<InputUnit *>(newUnit)) {
        _U_in.push_back(inu);
    }
}


// --- Construct

void InputLayer::construct(const int &nunits)
{
    for (int i = 1; i < nunits; ++i) {
        InputUnit * newUnit = new InputUnit(i - 1);
        _registerUnit(newUnit);
    }
}
