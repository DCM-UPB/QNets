#include "EuclideanDistanceMap.hpp"
#include "StringCodeUtilities.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>
#include <cmath>

#include <stdexcept>

// --- Helpers
double calcDistHelper(std::vector<double> srcv, const size_t ndim)
{
    double dist = 0.;
    for (size_t i=0; i<ndim; ++i) {
        dist += pow(srcv[i] - srcv[i+ndim], 2);
    }
    return sqrt(dist);
}

double EuclideanDistanceMap::_calcDist()
{
    double dist = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        dist += pow(_sources[i]->getValue() - _sources[i+_ndim]->getValue(), 2);
    }
    return sqrt(dist);
}


// --- Parameter manipulation

void EuclideanDistanceMap::setParameters(const size_t &ndim, const size_t &source_id1, const size_t &source_id2)
{
    std::vector<size_t> source_id0s = {source_id1, source_id2};
    MultiDimStaticMap::setParameters(ndim, source_id0s);
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
    for (size_t i=0; i<_ndim; ++i) {
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
    const double dist = _calcDist();

    if (dist > 0) {
        double d1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double d1v1 = _sources[i]->getFirstDerivativeValue(i1d);
            const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i1d);
            d1 += (v1 - v2) * (d1v1 - d1v2);
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
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double d1v1 = _sources[i]->getFirstDerivativeValue(i2d);
            const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i2d);
            const double d2v1 = _sources[i]->getSecondDerivativeValue(i2d);
            const double d2v2 = _sources[i+_ndim]->getSecondDerivativeValue(i2d);

            const double dv = (v1 -v2);
            const double d1d = d1v1 - d1v2;
            const double d2d = d2v1 - d2v2;
            d21 += dv * d2d + d1d*d1d;
            d22 += dv * d1d;
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
                const double v1 = _sources[i]->getValue();
                const double v2 = _sources[i+_ndim]->getValue();
                const double vdv1 = _sources[i]->getVariationalFirstDerivativeValue(iv1d);
                const double vdv2 = _sources[i+_ndim]->getVariationalFirstDerivativeValue(iv1d);
                vd1 += (v1 - v2) * (vdv1-vdv2);
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
