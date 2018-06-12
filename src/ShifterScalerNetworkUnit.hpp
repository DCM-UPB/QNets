#ifndef SHIFTER_SCALER_NETWORK_UNIT
#define SHIFTER_SCALER_NETWORK_UNIT

#include "NetworkUnit.hpp"
#include "StringCodeUtilities.hpp"

#include <string>

// Unit with linear output function applied after activation
class ShifterScalerNetworkUnit: virtual public NetworkUnit
{
private:
    void _applyShiftScale() { // meant to be called in computeValues
        _v = (_v + _shift) * _scale;
        if (_v1d) for (int i=0; i<_nx0; ++i) _v1d[i] *= _scale;
        if (_v2d) for (int i=0; i<_nx0; ++i) _v2d[i] *= _scale;
        if (_v1vd) for (int i=0; i<_nvp; ++i) _v1vd[i] *= _scale;
        if (_v1d1vd) for (int i=0; i<_nx0; ++i) for (int j=0; j<_nvp; ++j) _v1d1vd[i][j] *=_scale;
        if (_v2d1vd) for (int i=0; i<_nx0; ++i) for (int j=0; j<_nvp; ++j) _v2d1vd[i][j] *=_scale;
    }

protected:
    double _shift; // _shift will be added to the activation value
    double _scale; // and then _scale will be multiplied with the result to get the output value

public:
    // Constructor
    ShifterScalerNetworkUnit(const double shift = 0., const double scale = 1.)  {_shift = shift; _scale = scale;}

    // string code methods
    virtual std::string getIdCode(){return "ssu";} // return identifier for unit type
    virtual std::string getParams(){return composeParamCode("shift", _shift) + " , " + composeParamCode("scale", _scale);} // return parameter string

    virtual void setParams(const std::string &params){setParamValue(params, "shift", _shift); setParamValue(params, "scale", _scale);};

    // Setters
    void setShift(const double shift){_shift=shift;}
    void setScale(const double scale){_scale=scale;}

    // Getters
    double getShift() {return _shift;}
    double getScale() {return _scale;}

    void computeValues() {
        NetworkUnit::computeValues();
        _applyShiftScale();
    }
};

#endif
