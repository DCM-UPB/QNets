#include "EuclideanDistanceMap.hpp"
#include "StringCodeUtilities.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>
#include <cmath>

#include <stdexcept>

// --- Helpers
double calcDistHelper(std::vector<double> srcv, const int ndim)
{
    double dist = 0.;
    for (int i=0; i<ndim; ++i) {
        dist += pow(srcv[i] - srcv[i+ndim], 2);
    }
    return sqrt(dist);
}

double EuclideanDistanceMap::_calcDist()
{
    double dist = 0.;
    for (int i=0; i<_ndim; ++i) {
        dist += pow(_sources[i]->getValue() - _sources[i+_ndim]->getValue(), 2);
    }
    return sqrt(dist);
}


// --- Constructor

EuclideanDistanceMap::EuclideanDistanceMap(NetworkLayer * nl, const size_t &source_id1, const size_t &source_id2, const int ndim): _ndim(ndim)
{
    std::vector<size_t> source_ids;
    for (int i=0; i<_ndim; ++i) {
        source_ids.push_back(source_id1+i);
        source_ids.push_back(source_id2+i);
    }

    _fillSourcePool(nl);
    _fillSources(source_ids);
}


// --- StringCode methods

std::string EuclideanDistanceMap::getParams()
{
    std::string params = composeParamCode("ndim", _ndim);
    params = composeCodes(params, composeParamCode("source_id1", _source_ids[0]));
    params = composeCodes(params, composeParamCode("source_id2", _source_ids[_ndim]));
    return composeCodes(StaticFeeder::getParams(), params);
}


void EuclideanDistanceMap::setParams(const std::string &params)
{
    StaticFeeder::setParams(params);

    std::string str_ndim = readParamValue(params, "ndim");
    setParamValue(str_ndim, _ndim);

    size_t id1, id2;
    std::string str_id1 = readParamValue(params, "source_id1");
    setParamValue(str_id1, id1);
    std::string str_id2 = readParamValue(params, "source_id2");
    setParamValue(str_id2, id2);

    std::vector<size_t> ids;
    for (size_t i=0; i<(size_t)_ndim; ++i) {
        ids.push_back(id1+i);
        ids.push_back(id2+i);
    }

    _fillSources(ids);
    if (_vp_id_shift > -1) this->setVariationalParametersIndexes(_vp_id_shift, false);
}


// --- Parameter manipulation

void EuclideanDistanceMap::setParameters(const int &ndim, const size_t &source_id1, const size_t &source_id2)
{
    _ndim = ndim;

    std::vector<size_t> ids;
    for (size_t i=0; i<(size_t)_ndim; ++i) {
        ids.push_back(source_id1+i);
        ids.push_back(source_id2+i);
    }

    _fillSources(ids);
    if (_vp_id_shift > -1) this->setVariationalParametersIndexes(_vp_id_shift, false);
}


// --- Feed Mu and Sigma

double EuclideanDistanceMap::getFeedMu()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }
    return calcDistHelper(srcv, _ndim);
}


double EuclideanDistanceMap::getFeedSigma()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }
    double dist =  calcDistHelper(srcv, _ndim);

    double sigma = 0.;
    for (int i=0; i<_ndim; ++i) {
        sigma += pow(srcv[i] - srcv[i+_ndim], 2) * ( pow(_sources[i]->getOutputSigma(), 2) + pow(_sources[i+_ndim]->getOutputSigma(), 2) ); // (d²/dx² sigmaX)²
    }

    return sqrt(sigma) / dist;
}


// --- Computation


double EuclideanDistanceMap::getFeed()
{
    return _calcDist();
}


double EuclideanDistanceMap::getFirstDerivativeFeed(const int &i1d)
{
    double d1 = 0.;
    for (int i=0; i<_ndim; ++i) {
        const double v1 = _sources[i]->getValue();
        const double v2 = _sources[i+_ndim]->getValue();
        const double d1v1 = _sources[i]->getFirstDerivativeValue(i1d);
        const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i1d);
        d1 += (v1 - v2) * (d1v1 - d1v2);
    }

    return d1 / _calcDist();
}


double EuclideanDistanceMap::getSecondDerivativeFeed(const int &i2d)
{
    const double dist = _calcDist();
    const double dist2 = dist * dist;

    double d21 = 0.;
    double d22 = 0.;
    for (int i=0; i<_ndim; ++i) {
        const double v1 = _sources[i]->getValue();
        const double v2 = _sources[i+_ndim]->getValue();
        const double d1v1 = _sources[i]->getFirstDerivativeValue(i2d);
        const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i2d);
        const double d2v1 = _sources[i]->getSecondDerivativeValue(i2d);
        const double d2v2 = _sources[i+_ndim]->getSecondDerivativeValue(i2d);

        const double d1d = d1v1 - d1v2;
        d21 += (v1 - v2) * (d2v1 - d2v2) + d1d*d1d;
        d22 += (v1 - v2) * d1d;
    }

    return (d21 + d22*d22 / dist2) / dist;
}

double EuclideanDistanceMap::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        double vd1 = 0.;
        for (int i=0; i<_ndim; ++i) {
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double vdv1 = _sources[i]->getVariationalFirstDerivativeValue(iv1d);
            const double vdv2 = _sources[i+_ndim]->getVariationalFirstDerivativeValue(iv1d);
            vd1 += (v1 - v2) * (vdv1-vdv2);
        }

        return vd1 / _calcDist();
    }
    else return 0.;
}


double EuclideanDistanceMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        throw std::runtime_error("[EuclideanDistanceMap::getCrossFirstDerivativeFeed] Variational cross derivatives not supported yet.");
    }
    else return 0.;
}


double EuclideanDistanceMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d < _vp_id_shift) {
        throw std::runtime_error("[EuclideanDistanceMap::getCrossFirstDerivativeFeed] Variational cross derivatives not supported yet.");
    }
    else return 0.;
}
