#include "qnets/fmap/IdentityMap.hpp"

// --- fill/clear sources

void IdentityMap::_fillSources(const std::vector<size_t> &source_ids)
{
    OneDimStaticMap::_fillSources(source_ids);
    _src = _sources[0];
}

void IdentityMap::_clearSources()
{
    OneDimStaticMap::_clearSources();
    _src = nullptr;
}


// --- Parameter manipulation

void IdentityMap::setParameters(const size_t &source_id)
{
    std::vector<size_t> source_ids{source_id};
    OneDimStaticMap::setParameters(source_ids);
}

// --- Feed Mu and Sigma

double IdentityMap::getFeedMu()
{
    return _src->getOutputMu();
}


double IdentityMap::getFeedSigma()
{
    return _src->getOutputSigma();
}


// --- Computation


double IdentityMap::getFeed()
{
    return _src->getValue();
}


double IdentityMap::getFirstDerivativeFeed(const int &i1d)
{
    return _src->getFirstDerivativeValue(i1d);
}


double IdentityMap::getSecondDerivativeFeed(const int &i2d)
{
    return _src->getSecondDerivativeValue(i2d);
}


double IdentityMap::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        return _src->getVariationalFirstDerivativeValue(iv1d);
    }
    {
        return 0.;
    }
}


double IdentityMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d < _vp_id_shift) {
        return _src->getCrossFirstDerivativeValue(i1d, iv1d);
    }
    {
        return 0.;
    }
}


double IdentityMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d < _vp_id_shift) {
        return _src->getCrossSecondDerivativeValue(i2d, iv2d);
    }
    {
        return 0.;
    }
}
