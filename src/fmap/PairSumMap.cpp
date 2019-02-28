#include "ffnn/fmap/PairSumMap.hpp"
#include "ffnn/serial/StringCodeUtilities.hpp"

#include <cmath>
#include <string>
#include <vector>

// --- fill/clear sources

void PairSumMap::_fillSources(const std::vector<size_t> &source_ids)
{
    OneDimStaticMap::_fillSources(source_ids);
    _src0 = _sources[0];
    _src1 = _sources[1];
}

void PairSumMap::_clearSources()
{
    OneDimStaticMap::_clearSources();
    _src0 = nullptr;
    _src1 = nullptr;
}


// --- Parameter manipulation

void PairSumMap::setParameters(const size_t &source_id0, const size_t &source_id1)
{
    std::vector<size_t> source_ids { source_id0 , source_id1};
    OneDimStaticMap::setParameters(source_ids);
}

// --- Feed Mu and Sigma

double PairSumMap::getFeedMu()
{
    return _src0->getOutputMu() + _src1->getOutputMu();
}

double PairSumMap::getFeedSigma()
{
    return sqrt(pow(_src0->getOutputSigma(),2) + pow(_src1->getOutputSigma(),2));
}

// --- Computation


double PairSumMap::getFeed(){
    return _src0->getValue() + _src1->getValue();
}

double PairSumMap::getFirstDerivativeFeed(const int &i1d){
    return _src0->getFirstDerivativeValue(i1d) + _src1->getFirstDerivativeValue(i1d);
}

double PairSumMap::getSecondDerivativeFeed(const int &i2d){
    return _src0->getSecondDerivativeValue(i2d) + _src1->getSecondDerivativeValue(i2d);
}


double PairSumMap::getVariationalFirstDerivativeFeed(const int &iv1d){
    if (iv1d < _vp_id_shift) {
        return _src0->getVariationalFirstDerivativeValue(iv1d) + _src1->getVariationalFirstDerivativeValue(iv1d);
    }
     {return 0.;
}
}

double PairSumMap::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
    if (iv1d < _vp_id_shift) {
        return _src0->getCrossFirstDerivativeValue(i1d, iv1d) + _src1->getCrossFirstDerivativeValue(i1d, iv1d);
    }
     {return 0.;
}
}

double PairSumMap::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
    if (iv2d < _vp_id_shift) {
        return _src0->getCrossSecondDerivativeValue(i2d, iv2d) + _src1->getCrossSecondDerivativeValue(i2d, iv2d);
    }
     {return 0.;
}
}
