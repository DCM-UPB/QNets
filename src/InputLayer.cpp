#include "InputLayer.hpp"
#include "InputUnit.hpp"

// --- Constructor

InputLayer::InputLayer(const int &nunits)
{
    for (int i=1; i<nunits; ++i)
        {
            InputUnit * newUnit = new InputUnit(i-1);
            _U.push_back(newUnit);
            _U_in.push_back(newUnit);
        }
}

// --- Modify structure

void InputLayer::setSize(const int &nunits)
{
    for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
    _U_in.clear();

    _U.push_back(_U_off);
    for (int i=1; i<nunits; ++i)
        {
            InputUnit * newUnit = new InputUnit(i-1);
            _U.push_back(newUnit);
            _U_in.push_back(newUnit);
        }
}
