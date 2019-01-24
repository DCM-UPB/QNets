#include "InputLayer.hpp"
#include "InputUnit.hpp"

#include <array>
#include <algorithm>
#include <iostream>
// --- Register Unit

void InputLayer::_registerUnit(NetworkUnit * newUnit)
{
    NetworkLayer::_registerUnit(newUnit);
    if(InputUnit * inu = dynamic_cast<InputUnit *>(newUnit)) {
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

// --- String Code methods

std::string InputLayer::getParams()
{
    return composeCodes(NetworkLayer::getParams(), composeParamCode("sort", _doSort));
}

void InputLayer::setParams(const std::string &params)
{
    setParamValue(params, "sort", _doSort);
    NetworkLayer::setParams(params);
}

// computeValues with sorting

void InputLayer::computeValues()
{
    const int ninput = _U_in.size();
    int indices[ninput];
    double origPV[ninput];
    for (int i=0; i<ninput; ++i) {
        indices[i] = i;
        origPV[i] = _U_in[i]->getProtoValue();
    }

    if (_doSort) {
        std::sort(indices, indices+ninput, [&origPV](int i, int j) { // sort inputs by their values
                                               return origPV[i] < origPV[j];
                                           });
        for (int i=0; i<ninput; ++i) { // apply sort
            _U_in[i]->setInputIndex(indices[i]);
            _U_in[i]->setProtoValue(origPV[indices[i]]);
        }
    }

    NetworkLayer::computeValues(); // compute

    if (_doSort) {
        for (int i=0; i<ninput; ++i) { // revert sort
            _U_in[i]->setInputIndex(i);
            _U_in[i]->setProtoValue(origPV[i]);
        }
    }
}
