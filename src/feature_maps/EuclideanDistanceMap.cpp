#include "EuclideanDistanceMap.hpp"
#include "StringCodeUtilities.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>
#include <cmath>

#include <stdexcept>

// --- Helpers
double calcDistHelper(const std::vector<double> &srcv, const std::vector<double> &fixedPoint, const size_t &ndim)
{
    double dist = 0.;
    for (size_t i=0; i<ndim; ++i) {
        dist += pow(srcv[i] - fixedPoint[i], 2);
    }
    return dist;
}

double EuclideanDistanceMap::_calcDist()
{
    double dist = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        dist += pow(_sources[i]->getValue() - _fixedPoint[i], 2);
    }
    return dist;
}


// --- StringCode methods

std::string EuclideanDistanceMap::getParams()
{
    std::string params = MultiDimStaticMap::getParams();
    for (size_t i=0; i<_ndim; ++i) {
        params = composeCodes(params, composeParamCode("fp" + std::to_string(i), _fixedPoint[i]));
    }
    return params;
}


void EuclideanDistanceMap::setParams(const std::string &params)
{
    MultiDimStaticMap::setParams(params);

    _fixedPoint.clear();
    for (size_t i=0; i<_ndim; ++i) {
        double x;
        std::string str = readParamValue(params, "fp" + std::to_string(i));
        setParamValue(str, x);
        _fixedPoint.push_back(x);
    }
}

// --- Parameter manipulation

void EuclideanDistanceMap::setParameters(const size_t &ndim, const size_t &source_id0, const vector<double> &fixedPoint)
{
    std::vector<size_t> source_id0s = {source_id0};
    EuclideanDistanceMap::setParameters(ndim, source_id0s, fixedPoint);
}

void EuclideanDistanceMap::setParameters(const size_t &ndim, const vector<size_t> &source_id0s, const vector<double> &fixedPoint)
{
    _fixedPoint.clear();
    _fixedPoint = fixedPoint;
    MultiDimStaticMap::setParameters(ndim, source_id0s);
}


// --- Feed Mu and Sigma

double EuclideanDistanceMap::getFeedMu()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }
    return calcDistHelper(srcv, _fixedPoint, _ndim);
}


double EuclideanDistanceMap::getFeedSigma()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }

    double sigma = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        sigma += 4.0 * srcv[i]*srcv[i] * pow(_sources[i]->getOutputSigma(), 2); // (d²/dx² sigmaX)²
    }

    return sqrt(sigma);
}


// --- Computation


double EuclideanDistanceMap::getFeed()
{
    return _calcDist();
}


double EuclideanDistanceMap::getFirstDerivativeFeed(const int &i1d)
{
    double d1 = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        const double v = _sources[i]->getValue();
        const double d1v = _sources[i]->getFirstDerivativeValue(i1d);
        d1 += (v - _fixedPoint[i]) * d1v;
    }

    return 2.0 * d1;
}


double EuclideanDistanceMap::getSecondDerivativeFeed(const int &i2d)
{
    double d2 = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        const double v = _sources[i]->getValue();
        const double d1v = _sources[i]->getFirstDerivativeValue(i2d);
        const double d2v = _sources[i]->getSecondDerivativeValue(i2d);
        d2 += d1v*d1v + (v - _fixedPoint[i]) * d2v;
    }

    return 2.0 * d2;
}


double EuclideanDistanceMap::getVariationalFirstDerivativeFeed(const int &iv1d)

{
    if (iv1d < _vp_id_shift) {
        double vd1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v = _sources[i]->getValue();
            const double vdv = _sources[i]->getVariationalFirstDerivativeValue(iv1d);

            vd1 += (v - _fixedPoint[i]) * vdv;
        }

        return 2.0 * vd1;
    }

    return 0.;
}


double EuclideanDistanceMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        double cd1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v = _sources[i]->getValue();
            const double d1v = _sources[i]->getFirstDerivativeValue(i1d);
            const double vdv = _sources[i]->getVariationalFirstDerivativeValue(iv1d);
            const double cd1v = _sources[i]->getCrossFirstDerivativeValue(i1d, iv1d);

            cd1 += d1v * vdv + (v - _fixedPoint[i]) * cd1v;
        }

        return 2.0 * cd1;
    }

    else return 0.;
}


double EuclideanDistanceMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d < _vp_id_shift) {
        double cd2 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v = _sources[i]->getValue();
            const double d1v = _sources[i]->getFirstDerivativeValue(i2d);
            const double d2v = _sources[i]->getSecondDerivativeValue(i2d);
            const double vdv = _sources[i]->getVariationalFirstDerivativeValue(iv2d);
            const double cd1v = _sources[i]->getCrossFirstDerivativeValue(i2d, iv2d);
            const double cd2v = _sources[i]->getCrossFirstDerivativeValue(i2d, iv2d);

            cd2 += d2v * vdv + d1v * cd1v + d1v * cd1v + (v - _fixedPoint[i]) * cd2v;
        }

        return 2.0 * cd2;
    }

    else return 0.;
}
