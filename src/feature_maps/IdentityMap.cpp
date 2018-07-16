#include "IdentityMap.hpp"
#include "StringCodeUtilities.hpp"

#include <vector>
#include <string>

// --- Constructor

IdentityMap::IdentityMap(NetworkLayer * nl, const size_t &source_id)
{
    std::vector<size_t> source_ids = { source_id };
    _fillSourcePool(nl);
    _fillSources(source_ids);
}


// --- Feed Mu and Sigma

double IdentityMap::getFeedMu()
{
    return _sources[0]->getOutputMu();
}


double IdentityMap::getFeedSigma()
{
    return _sources[0]->getOutputSigma();
}


// --- Computation


double IdentityMap::getFeed(){
    return _sources[0]->getValue();
}


double IdentityMap::getFirstDerivativeFeed(const int &i1d){
    return _sources[0]->getFirstDerivativeValue(i1d);
}


double IdentityMap::getSecondDerivativeFeed(const int &i2d){
    return _sources[0]->getSecondDerivativeValue(i2d);
}


double IdentityMap::getVariationalFirstDerivativeFeed(const int &iv1d){
    if (iv1d < _vp_id_shift) {
        return _sources[0]->getVariationalFirstDerivativeValue(iv1d);
    }
    else return 0.;
}


double IdentityMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
    if (iv1d < _vp_id_shift) {
        return _sources[0]->getCrossFirstDerivativeValue(i1d, iv1d);
    }
    else return 0.;
}


double IdentityMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
    if (iv2d < _vp_id_shift) {
        return _sources[0]->getCrossSecondDerivativeValue(i2d, iv2d);
    }
    else return 0.;
}
