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
    return sqrt(dist);
}

double EuclideanDistanceMap::_calcDist()
{
    double dist = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        dist += pow(_sources[i]->getValue() - _fixedPoint[i], 2);
    }
    return sqrt(dist);
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
    double dist =  calcDistHelper(srcv, _fixedPoint, _ndim);

    double sigma = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        sigma += srcv[i]*srcv[i] * pow(_sources[i]->getOutputSigma(), 2); // (d²/dx² sigmaX)²
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
    const double dist = _calcDist();

    if (dist > 0) {
        double d1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v = _sources[i]->getValue();
            const double d1v = _sources[i]->getFirstDerivativeValue(i1d);
            d1 += (v - _fixedPoint[i]) * d1v;
        }

        return d1 / dist;
    }
    else return 0.;
}


double EuclideanDistanceMap::getSecondDerivativeFeed(const int &i2d)
{
    const double dist = _calcDist();

    if (dist > 0) {
        const double dist2 = dist * dist;

        double d21 = 0.;
        double d22 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v = _sources[i]->getValue();
            const double d1v = _sources[i]->getFirstDerivativeValue(i2d);
            const double d2v = _sources[i]->getSecondDerivativeValue(i2d);

            const double dv = v - _fixedPoint[i];
            d21 += dv * d2v + d1v*d1v;
            d22 += dv * d1v;
        }

        return (d21 - d22*d22 / dist2) / dist;
    }
    else return 0.;
}

double EuclideanDistanceMap::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        const double dist = _calcDist();

        if (dist > 0) {
            double vd1 = 0.;
            for (size_t i=0; i<_ndim; ++i) {
                const double v = _sources[i]->getValue();
                const double vdv = _sources[i]->getVariationalFirstDerivativeValue(iv1d);

                vd1 += (v - _fixedPoint[i]) * vdv;
            }

            return vd1 / dist;
        }
    }
    return 0.;
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
