#include "NNUnit.hpp"

// --- Computation

void NNUnit::computeActivation(){
    const bool flag_d1 = _v1d || _v2d || _v1vd || _v1d1vd;
    const bool flag_d2 = _v2d || _v1vd;
    const bool flag_d3 = _v2d1vd;
    _actf->fad(_pv, _v, _a1d, _a2d, _a3d, flag_d1, flag_d2, flag_d3);
}

