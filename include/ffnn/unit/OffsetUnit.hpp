#ifndef FFNN_UNIT_OFFSETUNIT_HPP
#define FFNN_UNIT_OFFSETUNIT_HPP

#include "ffnn/unit/NetworkUnit.hpp"

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
    double getIdealProtoMu() override { return 1.; }
    double getIdealProtoSigma() override { return 0.; }

    // there is no variation in the offset
    double getOutputMu() override { return _v; }
    double getOutputSigma() override { return 0.; }

    // string code methods
    std::string getIdCode() override { return "OFF"; } // return identifier for unit type

    // Computation
    void computeFeed() override {}
    void computeActivation() {}
    void computeDerivatives() override {}

    void computeValues() override { _v = _pv; }
};

#endif
