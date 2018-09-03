#include "EuclideanPairDistanceMap.hpp"
#include "StringCodeUtilities.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>
#include <cmath>

#include <stdexcept>

// --- Helpers
double calcDistHelper(std::vector<double> &srcv, const size_t &ndim)
{
    double dist = 0.;
    for (size_t i=0; i<ndim; ++i) {
        dist += pow(srcv[i] - srcv[i+ndim], 2);
    }
    return dist;
}

double EuclideanPairDistanceMap::_calcDist()
{
    double dist = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        dist += pow(_sources[i]->getValue() - _sources[i+_ndim]->getValue(), 2);
    }
    return dist;
}


// --- Parameter manipulation

void EuclideanPairDistanceMap::setParameters(const size_t &ndim, const size_t &source_id0, const size_t &source_id1)
{
    std::vector<size_t> source_id0s = {source_id0, source_id1};
    MultiDimStaticMap::setParameters(ndim, source_id0s);
}


// --- Feed Mu and Sigma

double EuclideanPairDistanceMap::getFeedMu()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }
    return calcDistHelper(srcv, _ndim);
}


double EuclideanPairDistanceMap::getFeedSigma()
{
    std::vector<double> srcv;
    for (size_t i=0; i<_sources.size(); ++i) {
        srcv.push_back(_sources[i]->getOutputMu());
    }

    double sigma = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        sigma += 2.0 * pow(srcv[i] - srcv[i+_ndim], 2) * ( pow(_sources[i]->getOutputSigma(), 2) + pow(_sources[i+_ndim]->getOutputSigma(), 2) ); // (d²/dx² sigmaX)²
    }

    return sqrt(sigma);
}


// --- Computation


double EuclideanPairDistanceMap::getFeed()
{
    return _calcDist();
}


double EuclideanPairDistanceMap::getFirstDerivativeFeed(const int &i1d)
{
    double d1 = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        const double v1 = _sources[i]->getValue();
        const double v2 = _sources[i+_ndim]->getValue();
        const double d1v1 = _sources[i]->getFirstDerivativeValue(i1d);
        const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i1d);
        d1 += (v1 - v2) * (d1v1 - d1v2);
    }

    return 2.0 * d1;
}


double EuclideanPairDistanceMap::getSecondDerivativeFeed(const int &i2d)
{
    double d2 = 0.;
    for (size_t i=0; i<_ndim; ++i) {
        const double v1 = _sources[i]->getValue();
        const double v2 = _sources[i+_ndim]->getValue();
        const double d1v1 = _sources[i]->getFirstDerivativeValue(i2d);
        const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i2d);
        const double d2v1 = _sources[i]->getSecondDerivativeValue(i2d);
        const double d2v2 = _sources[i+_ndim]->getSecondDerivativeValue(i2d);

        const double diff = v1 - v2;
        const double d1diff = d1v1 - d1v2;
        const double d2diff = d2v1 - d2v2;

        d2 += d1diff * d1diff + diff * d2diff;
    }

    return 2.0 * d2;
}

double EuclideanPairDistanceMap::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        double vd1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double vdv1 = _sources[i]->getVariationalFirstDerivativeValue(iv1d);
            const double vdv2 = _sources[i+_ndim]->getVariationalFirstDerivativeValue(iv1d);

            vd1 += (v1 - v2) * (vdv1 - vdv2);
        }

        return 2.0 * vd1;
    }

    return 0.;
}


double EuclideanPairDistanceMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        double cd1 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double d1v1 = _sources[i]->getFirstDerivativeValue(i1d);
            const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i1d);
            const double vdv1 = _sources[i]->getVariationalFirstDerivativeValue(iv1d);
            const double vdv2 = _sources[i+_ndim]->getVariationalFirstDerivativeValue(iv1d);
            const double cd1v1 = _sources[i]->getCrossFirstDerivativeValue(i1d, iv1d);
            const double cd1v2 = _sources[i+_ndim]->getCrossFirstDerivativeValue(i1d, iv1d);

            const double diff = v1 - v2;
            const double d1diff = d1v1 - d1v2;
            const double vddiff = vdv1 - vdv2;
            const double cd1diff = cd1v1 - cd1v2;

            cd1 += d1diff * vddiff + diff * cd1diff;
        }

        return 2.0 * cd1;
    }

    else return 0.;
}


double EuclideanPairDistanceMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d < _vp_id_shift) {
        double cd2 = 0.;
        for (size_t i=0; i<_ndim; ++i) {
            const double v1 = _sources[i]->getValue();
            const double v2 = _sources[i+_ndim]->getValue();
            const double d1v1 = _sources[i]->getFirstDerivativeValue(i2d);
            const double d1v2 = _sources[i+_ndim]->getFirstDerivativeValue(i2d);
            const double d2v1 = _sources[i]->getSecondDerivativeValue(i2d);
            const double d2v2 = _sources[i+_ndim]->getSecondDerivativeValue(i2d);
            const double vdv1 = _sources[i]->getVariationalFirstDerivativeValue(iv2d);
            const double vdv2 = _sources[i+_ndim]->getVariationalFirstDerivativeValue(iv2d);
            const double cd1v1 = _sources[i]->getCrossFirstDerivativeValue(i2d, iv2d);
            const double cd1v2 = _sources[i+_ndim]->getCrossFirstDerivativeValue(i2d, iv2d);
            const double cd2v1 = _sources[i]->getCrossSecondDerivativeValue(i2d, iv2d);
            const double cd2v2 = _sources[i+_ndim]->getCrossSecondDerivativeValue(i2d, iv2d);

            const double diff = v1 - v2;
            const double d1diff = d1v1 - d1v2;
            const double d2diff = d2v1 - d2v2;
            const double vddiff = vdv1 - vdv2;
            const double cd1diff = cd1v1 - cd1v2;
            const double cd2diff = cd2v1 - cd2v2;

            cd2 += d2diff * vddiff + 2.0 * d1diff * cd1diff + diff * cd2diff;
        }

        return 2.0 * cd2;
    }

    else return 0.;
}
