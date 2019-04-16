#ifndef FFNN_UNIT_INPUTUNIT_HPP
#define FFNN_UNIT_INPUTUNIT_HPP

#include "ffnn/unit/ShifterScalerUnit.hpp"
#include <string>

// Input Unit
class InputUnit: public ShifterScalerUnit
{
protected:
    const int _index;
    double _inputMu, _inputSigma;

public:
    explicit InputUnit(const int &index, const double inputMu = 0., const double inputSigma = 1.):
            _index(index)
    {
        _inputMu = inputMu;
        _inputSigma = inputSigma;
    } // the index of the input unit, i.e. d/dx_index f(_pv) = 1
    ~InputUnit() override = default;

    // string code methods
    std::string getIdCode() override { return "IN"; } // return identifier for unit type

    // return the output mean value (mu) and standard deviation (sigma)
    double getOutputMu() override { return (_inputMu + _shift)*_scale; }
    double getOutputSigma() override { return _inputSigma*_scale; }

    // set input data mu and sigma, set shift/scale accordingly
    void setInputMu(const double &inputMu, const bool &doShift = true);
    void setInputSigma(const double &inputSigma, const bool &doScale = true);

    // get the input data mu and sigma
    double getInputMu() { return _inputMu; }
    double getInputSigma() { return _inputSigma; }

    // Computation
    void computeFeed() override {}
    void computeActivation() {}
    void computeDerivatives() override {}

    void computeValues() override
    {
        _v = _pv;
        if (_v1d != nullptr) {
            _v1d[_index] = 1.;
        }
    }
};


#endif
