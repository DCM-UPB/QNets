#ifndef FFNN_UNIT_OFFSETUNIT_HPP
#define FFNN_UNIT_OFFSETUNIT_HPP

#include "qnets/poly/unit/NetworkUnit.hpp"

// Offset Unit
class OffsetUnit: public NetworkUnit
{
public:
    OffsetUnit()
    {
        _pv = 1.;
        _v = 1.;
    }

    // there is no real ideal mu / sigma for offset, we return the closest thing
    double getIdealProtoMu() final { return 1.; }
    double getIdealProtoSigma() final { return 0.; }

    // there is no variation in the offset
    double getOutputMu() final { return _v; }
    double getOutputSigma() final { return 0.; }

    // string code methods
    std::string getIdCode() final { return "OFF"; } // return identifier for unit type

    // Computation
    void computeFeed() final {}
    void computeActivation() {}
    void computeDerivatives() final {}

    void computeValues() final { _v = _pv; }
};

#endif
