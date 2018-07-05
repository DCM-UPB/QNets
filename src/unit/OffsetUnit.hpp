#ifndef OFFSET_UNIT
#define OFFSET_UNIT

#include "NetworkUnit.hpp"

// Offset Unit
class OffsetUnit: public NetworkUnit
{
public:
    OffsetUnit(){_pv = 1.; _v = 1.;}

    // there is no real ideal mu / sigma for offset, we return the closest thing
    virtual double getIdealProtoMu(){return _pv;}
    virtual double getIdealProtoSigma(){return 0.;}

    // there is no variation in the offset
    virtual double getOutputMu(){return _v;}
    virtual double getOutputSigma(){return 0.;}

    // string code methods
    virtual std::string getIdCode(){return "OFF";} // return identifier for unit type

    // Computation
    void computeFeed(){}
    void computeActivation(){}
    void computeDerivatives(){}

    void computeValues() {_v = _pv;}
};

#endif
