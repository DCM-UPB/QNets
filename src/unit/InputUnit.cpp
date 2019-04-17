#include "qnets/unit/InputUnit.hpp"

// set input data mu and sigma, set shift/scale accordingly
void InputUnit::setInputMu(const double &inputMu, const bool &doShift)
{
    _inputMu = inputMu;
    if (doShift) {
        _shift = -inputMu;
    }
}

void InputUnit::setInputSigma(const double &inputSigma, const bool &doScale)
{
    _inputSigma = inputSigma;
    if (doScale) {
        inputSigma > 0 ? _scale = 1./inputSigma : _scale = 1.;
    }
}
