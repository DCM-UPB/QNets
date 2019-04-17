#include "qnets/fmap/PairDifferenceMap.hpp"

#include <cmath>

// --- fill/clear sources

void PairDifferenceMap::_fillSources(const std::vector<size_t> &source_ids)
{
    OneDimStaticMap::_fillSources(source_ids);
    _src0 = _sources[0];
    _src1 = _sources[1];
}

void PairDifferenceMap::_clearSources()
{
    OneDimStaticMap::_clearSources();
    _src0 = nullptr;
    _src1 = nullptr;
}


// --- Parameter manipulation

void PairDifferenceMap::setParameters(const size_t &source_id0, const size_t &source_id1)
{
    std::vector<size_t> source_ids{source_id0, source_id1};
    OneDimStaticMap::setParameters(source_ids);
}


// --- Feed Mu and Sigma

double PairDifferenceMap::getFeedMu()
{
    return _src0->getOutputMu() - _src1->getOutputMu();
}

double PairDifferenceMap::getFeedSigma()
{
    return sqrt(pow(_src0->getOutputSigma(), 2) + pow(_src1->getOutputSigma(), 2));
}


// --- Computation


double PairDifferenceMap::getFeed()
{
    return _src0->getValue() - _src1->getValue();
}

double PairDifferenceMap::getFirstDerivativeFeed(const int &i1d)
{
    return _src0->getFirstDerivativeValue(i1d) - _src1->getFirstDerivativeValue(i1d);
}

double PairDifferenceMap::getSecondDerivativeFeed(const int &i2d)
{
    return _src0->getSecondDerivativeValue(i2d) - _src1->getSecondDerivativeValue(i2d);
}


double PairDifferenceMap::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        return _src0->getVariationalFirstDerivativeValue(iv1d) - _src1->getVariationalFirstDerivativeValue(iv1d);
    }
    {
        return 0.;
    }
}

double PairDifferenceMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        return _src0->getCrossFirstDerivativeValue(i1d, iv1d) - _src1->getCrossFirstDerivativeValue(i1d, iv1d);
    }
    {
        return 0.;
    }
}

double PairDifferenceMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d < _vp_id_shift) {
        return _src0->getCrossSecondDerivativeValue(i2d, iv2d) - _src1->getCrossSecondDerivativeValue(i2d, iv2d);
    }
    {
        return 0.;
    }
}
