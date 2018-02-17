#include "NNUnit.hpp"

#include <iostream>

// --- Computation

void NNUnit::computeValues(){
    if (_feeder){
        // unit value
        _pv = _feeder->getFeed();
        _v = _actf->f(_pv);
        // shared useful values
        double a1d = _actf->f1d(_pv);
        if (_v1d || _v2d){
            for (int i=0; i<_nx0; ++i)
            {
                _fdf[i] = _feeder->getFirstDerivativeFeed(i);
            }
        }
        // first derivative
        if (_v1d){
            for (int i=0; i<_nx0; ++i)
            {
                _v1d[i] = a1d * _fdf[i];
            }
        }
        // second derivative
        if (_v2d){
            double a2d = _actf->f2d(_pv);
            for (int i=0; i<_nx0; ++i)
            {
                _v2d[i] = a1d * _feeder->getSecondDerivativeFeed(i) +
                a2d * _fdf[i] * _fdf[i];
            }
        }
        // variational first derivative
        if (_v1vd){
            for (int i=0; i<_nvp; ++i)
            {
                _v1vd[i] = a1d * _feeder->getVariationalFirstDerivativeFeed(i);
            }
        }
        // cross first derivative
        if (_v1d1vd){
            for (int i=0; i<_nx0; ++i){
                for (int j=0; j<_nvp; ++j){
                    _v1d1vd[i][j] = a1d * _feeder->getCrossFirstDerivativeFeed(i, j);
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

    if (!_fdf)
    {
        _fdf = new double[nx0];
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

    if (!_fdf)
    {
        _fdf = new double[nx0];
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
}


// --- Cross Variational/Coordinate derivatives

void NNUnit::setCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
{
    _nx0 = nx0;
    _nvp = nvp;
    _v1d1vd = new double*[_nx0];
    for (int i=0; i<_nx0; ++i){
        _v1d1vd[i] = new double[_nvp];
        for (int j=0; j<_nvp; ++j){
            _v1d1vd[i][j] = 0.;
        }
    }
    // TODO: perhaps other derivative substrates are necessary??
}


// --- Constructor

NNUnit::NNUnit(ActivationFunctionInterface * actf)
{
    _actf = actf;
    _pv = 0.;
    _v = 0.;
    _feeder = NULL;
    _v1d = NULL;
    _v2d = NULL;
    _fdf = NULL;
    _v1vd = NULL;
    _v1d1vd = NULL;
}

// --- Destructor

NNUnit::~NNUnit()
{
    if (_feeder)
    {
        delete _feeder;
    }

    if (_v1d)
    {
        delete[] _v1d;
    }

    if (_v2d)
    {
        delete[] _v2d;
    }

    if (_fdf)
    {
        delete[] _fdf;
    }

    if (_v1vd)
    {
        delete[] _v1vd;
    }
}
