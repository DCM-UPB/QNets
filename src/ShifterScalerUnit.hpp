#ifndef SHIFTER_SCALER_UNIT
#define SHIFTER_SCALER_UNIT

#include "NetworkUnit.hpp"
#include "NetworkUnitFeederInterface.hpp"


// Unit with linear output function applied after activation
class ShifterScalerUnit: virtual public NetworkUnit
{
protected:
    double _shift; // _shift will be added to the activation value
    double _scale; // and then _scale will be multiplied with the result to get the output value

    void _applyShiftScale() { // meant to be called in computeValues
        _v = (_v + _shift) * _scale;
        if (_v1d) for (int i=0; i<_nx0; ++i) _v1d[i] *= _scale;
        if (_v2d) for (int i=0; i<_nx0; ++i) _v2d[i] *= _scale;
        if (_v1vd) for (int i=0; i<_nvp; ++i) _v1vd[i] *= _scale;
        if (_v1d1vd) for (int i=0; i<_nx0; ++i) for (int j=0; j<_nvp; ++j) _v1d1vd[i][j] *=_scale;
        if (_v2d1vd) for (int i=0; i<_nx0; ++i) for (int j=0; j<_nvp; ++j) _v2d1vd[i][j] *=_scale;
    }

public:
    // Constructor
    ShifterScalerUnit(NetworkUnitFeederInterface * feeder = NULL, const double shift = 0., const double scale = 1.) : NetworkUnit(feeder) {_shift = shift; _scale = scale;}

    // Setters
    void setShift(const double shift){_shift=shift;}
    void setScale(const double scale){_scale=scale;}

    // Getters
    double getShift() {return _shift;}
    double getScale() {return _scale;}

    virtual void computeValues() {
        NetworkUnit::computeValues();
        _applyShiftScale();
    }
};

#endif
