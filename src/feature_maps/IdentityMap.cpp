#include "IdentityMap.hpp"
#include "StringCodeUtilities.hpp"

#include <vector>
#include <string>
#include <iostream>
// --- Constructor

IdentityMap::IdentityMap(NetworkLayer * nl, const size_t &source_id)
{
    _fillSourcePool(nl);
    setParameters(source_id);
}


// --- StringCode methods

std::string IdentityMap::getParams()
{
    return composeCodes(StaticFeeder::getParams(), composeParamCode("source_id", _source_ids[0]));
}


void IdentityMap::setParams(const std::string &params)
{
    std::vector<size_t> source_ids(1);
    setParamValue(readParamValue(params, "source_id"), source_ids[0]);
    _fillSources(source_ids);
    StaticFeeder::setParams(params);
}


// --- Parameter manipulation

void IdentityMap::setParameters(const size_t &source_id)
{
    std::vector<size_t> source_ids { source_id };
    _fillSources(source_ids);
    if (_vp_id_shift > -1) this->setVariationalParametersIndexes(_vp_id_shift, false);
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
