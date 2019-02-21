#include "ffnn/unit/FedUnit.hpp"
#include "ffnn/serial/StringCodeUtilities.hpp"

#include <string>
#include <cstddef> // for NULL


// --- Computation

void FedUnit::computeFeed(){
    if (_feeder){
        // unit value
        _pv = _feeder->getFeed();

        // feed derivatives
        if (_first_der){
            for (int i=0; i<_nx0; ++i) _first_der[i] = _feeder->getFirstDerivativeFeed(i);
        }

        if (_second_der){
            for (int i=0; i<_nx0; ++i) _second_der[i] = _feeder->getSecondDerivativeFeed(i);
        }

        if (_first_var_der) {
            for (int j=0; j<_nvp; ++j) _first_var_der[j] = _feeder->getVariationalFirstDerivativeFeed(j);
        }

        if (_cross_first_der) {
            for (int j=0; j<_nvp; ++j) {
                for (int i=0; i<_nx0; ++i) _cross_first_der[i][j] = _feeder->getCrossFirstDerivativeFeed(i, j);
            }
        }

        if (_cross_second_der) {
            for (int j=0; j<_nvp; ++j) {
                for (int i=0; i<_nx0; ++i) _cross_second_der[i][j] = _feeder->getCrossSecondDerivativeFeed(i, j);
            }
        }
    }
}


void FedUnit::computeDerivatives(){
    if (_feeder) {
        // first derivative
        if (_v1d){
            for (int i=0; i<_nx0; ++i)
                {
                    _v1d[i] = _a1d * _first_der[i];
                }
        }
        // second derivative
        if (_v2d){
            for (int i=0; i<_nx0; ++i)
                {
                    _v2d[i] = _a1d * _second_der[i] + _a2d * _first_der[i] * _first_der[i];
                }
        }
        // variational first derivative
        if (_v1vd){
            for (int i=0; i<_nvp; ++i)
                {
                    _v1vd[i] = _a1d * _first_var_der[i];
                }
        }
        // cross first derivative
        if (_v1d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j){
                    _v1d1vd[i][j] = 0.;
                    _v1d1vd[i][j] += _a1d * _cross_first_der[i][j];
                    _v1d1vd[i][j] += _a2d * _first_der[i] * _first_var_der[j];
                }
            }
        }
        // cross second derivative
        if (_v2d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j){
                    _v2d1vd[i][j] = 0.;
                    _v2d1vd[i][j] += _a1d * _cross_second_der[i][j];
                    _v2d1vd[i][j] += 2. * _a2d * _first_der[i] * _cross_first_der[i][j];
                    _v2d1vd[i][j] += _a3d * _first_der[i] * _first_der[i] * _first_var_der[j];
                    _v2d1vd[i][j] += _a2d * _second_der[i] * _first_var_der[j];
                }
            }
        }
    }
}
