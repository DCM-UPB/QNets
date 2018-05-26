#include "NNUnit.hpp"

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



// --- Coordinate derivatives

void NNUnit::setFirstDerivativeSubstrate(const int &nx0)
{
    _nx0 = nx0;
    _v1d = new double[_nx0];
    for (int i=0; i<_nx0; ++i)
        {
            _v1d[i]=0.;
        }

    if (!_first_der){
        _first_der = new double[nx0];
    }
}


void NNUnit::setSecondDerivativeSubstrate(const int &nx0)
{
    _nx0 = nx0;
    _v2d = new double[_nx0];
    for (int i=0; i<_nx0; ++i)
        {
            _v2d[i]=0.;
        }

    if (!_first_der){
        _first_der = new double[nx0];
    }

    if (!_second_der){
        _second_der = new double[nx0];
    }
}


// --- Variational derivatives

void NNUnit::setVariationalFirstDerivativeSubstrate(const int &nvp)
{
    _nvp = nvp;
    _v1vd = new double[_nvp];
    for (int i=0; i<_nvp; ++i)
        {
            _v1vd[i]=0.;
        }

    if (!_first_var_der){
        _first_var_der = new double[nvp];
    }
}


// --- Cross Variational/Coordinate derivatives

void NNUnit::setCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp){
    _nx0 = nx0;
    _nvp = nvp;
    _v1d1vd = new double*[_nx0];
    for (int i=0; i<_nx0; ++i){
        _v1d1vd[i] = new double[_nvp];
        for (int j=0; j<_nvp; ++j){
            _v1d1vd[i][j] = 0.;
        }
    }

    if (!_first_var_der){
        _first_var_der = new double[nvp];
    }

    if (!_cross_first_der){
        _cross_first_der = new double*[nx0];
        for (int i=0; i<nx0; ++i) _cross_first_der[i] = new double[nvp];
    }
}


void NNUnit::setCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp){
    _nx0 = nx0;
    _nvp = nvp;
    _v2d1vd = new double*[_nx0];
    for (int i=0; i<_nx0; ++i){
        _v2d1vd[i] = new double[_nvp];
        for (int j=0; j<_nvp; ++j){
            _v2d1vd[i][j] = 0.;
        }
    }

    if (!_first_var_der){
        _first_var_der = new double[nvp];
    }

    if (!_cross_first_der){
        _cross_first_der = new double*[nx0];
        for (int i=0; i<nx0; ++i) _cross_first_der[i] = new double[nvp];
    }

    if (!_second_der){
        _second_der = new double[nx0];
    }
}


// --- Constructor

NNUnit::NNUnit(ActivationFunctionInterface * actf){
    _actf = actf;
    _pv = 0.;
    _v = 0.;
    _feeder = NULL;
    _v1d = NULL;
    _v2d = NULL;
    _first_der = NULL;
    _second_der = NULL;
    _first_var_der = NULL;
    _cross_first_der = NULL;
    _v1vd = NULL;
    _v1d1vd = NULL;
    _v2d1vd = NULL;
}

// --- Destructor

NNUnit::~NNUnit(){
    if (_feeder) delete _feeder;
    if (_v1d) delete[] _v1d;
    if (_v2d) delete[] _v2d;
    if (_first_der) delete[] _first_der;
    if (_second_der) delete[] _second_der;
    if (_first_var_der) delete[] _first_var_der;
    if (_v1vd) delete[] _v1vd;
    if (_v1d1vd){
        for (int i=0; i<_nx0; ++i){
            delete[] _v1d1vd[i];
        }
        delete[] _v1d1vd;
    }
    if (_v2d1vd){
        for (int i=0; i<_nx0; ++i){
            delete[] _v2d1vd[i];
        }
        delete[] _v2d1vd;
    }
    if (_cross_first_der){
        for (int i=0; i<_nx0; ++i) delete[] _cross_first_der[i];
        delete[] _cross_first_der;
    }
}
