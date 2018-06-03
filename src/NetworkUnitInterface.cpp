#include "NetworkUnitInterface.hpp"

#include <iostream>


// --- Coordinate derivatives

void NetworkUnitInterface::setFirstDerivativeSubstrate(const int &nx0)
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


void NetworkUnitInterface::setSecondDerivativeSubstrate(const int &nx0)
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

void NetworkUnitInterface::setVariationalFirstDerivativeSubstrate(const int &nvp)
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

void NetworkUnitInterface::setCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp){
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


void NetworkUnitInterface::setCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp){
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

NetworkUnitInterface::NetworkUnitInterface(){
    _pv = 0.;
    _v = 0.;
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

NetworkUnitInterface::~NetworkUnitInterface(){
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
