#include "NNUnit.hpp"
#include "NNUnitFeederInterface.hpp"

#include <iostream>

// --- Computation

void NNUnit::computeValues(){
    if (_feeder){
        // unit value
        _pv = _feeder->getFeed();
        // shared useful values
        double a1d, a2d, a3d;

        const bool flag_d1 = _v1d || _v2d || _v1vd || _v1d1vd;
        const bool flag_d2 = _v2d || _v1vd;
        const bool flag_d3 = _v2d1vd;
        _actf->fad(_pv, _v, a1d, a2d, a3d, flag_d1, flag_d2, flag_d3);

        if (_v1d || _v2d){
            for (int i=0; i<_nx0; ++i) _first_der[i] = _feeder->getFirstDerivativeFeed(i);
        }

        if (_v2d || _v2d1vd){
            for (int i=0; i<_nx0; ++i) _second_der[i] = _feeder->getSecondDerivativeFeed(i);
        }

        if (_v1vd || _v1d1vd || _v2d1vd){
            for (int i=0; i<_nvp; ++i) _first_var_der[i] = _feeder->getVariationalFirstDerivativeFeed(i);
        }

        if (_v1d1vd || _v2d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j) _cross_first_der[i][j] = _feeder->getCrossFirstDerivativeFeed(i, j);
            }
        }

        // first derivative
        if (_v1d){
            for (int i=0; i<_nx0; ++i)
                {
                    _v1d[i] = a1d * _first_der[i];
                }
        }
        // second derivative
        if (_v2d){
            for (int i=0; i<_nx0; ++i)
                {
                    _v2d[i] = a1d * _second_der[i] + a2d * _first_der[i] * _first_der[i];
                }
        }
        // variational first derivative
        if (_v1vd){
            for (int i=0; i<_nvp; ++i)
                {
                    _v1vd[i] = a1d * _first_var_der[i];
                }
        }
        // cross first derivative
        if (_v1d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j){
                    _v1d1vd[i][j] = 0.;
                    if (_feeder->isBetaIndexUsedForThisRay(j)){
                        _v1d1vd[i][j] += a1d * _cross_first_der[i][j];
                        _v1d1vd[i][j] += a2d * _first_der[i] * _first_var_der[j];
                    }
                }
            }
        }
        // cross second derivative
        if (_v2d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j){
                    _v2d1vd[i][j] = 0.;
                    if (_feeder->isBetaIndexUsedForThisRay(j)){
                        _v2d1vd[i][j] += a1d * _feeder->getCrossSecondDerivativeFeed(i, j);
                        _v2d1vd[i][j] += 2. * a2d * _first_der[i] * _cross_first_der[i][j];
                        _v2d1vd[i][j] += a3d * _first_der[i] * _first_der[i] * _first_var_der[j];
                        _v2d1vd[i][j] += a2d * _second_der[i] * _first_var_der[j];
                    }
                }
            }
        }
    }
    else{
        _v = _actf->f(_pv);
    }
}

// --- Constructor

NNUnit::NNUnit(ActivationFunctionInterface * actf, NNUnitFeederInterface * feeder){
    _actf = actf;
    _feeder = feeder;
}

// --- Destructor

NNUnit::~NNUnit(){
    if (_feeder) delete _feeder; // actually feeder shouldnt be deleted here because it is an external allocation..
}
