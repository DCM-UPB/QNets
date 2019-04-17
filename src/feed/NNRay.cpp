#include "qnets/feed/NNRay.hpp"

#include <cmath>
#include <random>

// --- Constructor

NNRay::NNRay(NetworkLayer * nl)
{
    _fillSourcePool(nl);
    _fillSources(); // select all sources
    _fillBeta(); // one beta per source
    randomizeBeta();
}

// --- final setParams

void NNRay::setParams(const std::string &params)
{
    WeightedFeeder::setParams(params); // we don't need more to init vp system
    if (_vp_id_shift > -1) {
        setVariationalParametersIndexes(_vp_id_shift, _flag_vp);
    }
}

int NNRay::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp)
{
    return WeightedFeeder::setVariationalParametersIndexes(starting_index, flag_add_vp);
}


// --- Feed Mu and Sigma

double NNRay::getFeedMu()
{
    double mu = 0.;
    for (std::vector<NetworkUnit *>::size_type i = 0; i < _sources.size(); ++i) {
        mu += _beta[i]*_sources[i]->getOutputMu();
    }
    return mu;
}


double NNRay::getFeedSigma()
{
    double var = 0.;
    for (std::vector<NetworkUnit *>::size_type i = 0; i < _sources.size(); ++i) {
        var += pow(_beta[i]*_sources[i]->getOutputSigma(), 2);
    }
    return sqrt(var);
}


// --- Randomizers

void NNRay::randomizeBeta()
{
    // random number generator, used to initialize the intensities
    std::random_device rdev;
    std::mt19937_64 rgen;
    std::uniform_real_distribution<double> rd;

    // target sigma to keep sum of weighted inputs in range [-4,4], assuming uniform distribution
    // sigma = 8/sqrt(12) = (b-a)/sqrt(12) * m^(1/2)
    const double bah = 4*pow(_sourcePool.size(), -0.5); // (b-a)/2

    rgen = std::mt19937_64(rdev());
    rd = std::uniform_real_distribution<double>(-bah, bah);

    for (double &b : _beta) {
        b = rd(rgen);
    }
}



// --- Computation


double NNRay::getFeed()
{
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i = 0; i < _sources.size(); ++i) {
        feed += _beta[i]*_sources[i]->getValue();
    }
    return feed;
}


double NNRay::getFirstDerivativeFeed(const int &i1d)
{
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i = 1; i < _sources.size(); ++i) {
        feed += _beta[i]*_sources[i]->getFirstDerivativeValue(i1d);
    }

    return feed;
}


double NNRay::getSecondDerivativeFeed(const int &i2d)
{
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i = 1; i < _sources.size(); ++i) {
        feed += _beta[i]*_sources[i]->getSecondDerivativeValue(i2d);
    }
    return feed;
}


double NNRay::getVariationalFirstDerivativeFeed(const int &iv1d)
{
    if (iv1d >= _vp_id_shift) {
        // if the variational parameter with index iv1d is in the ray add the following element
        return _sources[iv1d - _vp_id_shift]->getValue();
    }

    // else add source components
    double feed = 0.;
    for (size_t i = 0; i < _map_index_to_sources[iv1d].size(); ++i) {
        feed += _beta[_map_index_to_sources[iv1d][i]]*_sources[_map_index_to_sources[iv1d][i]]->getVariationalFirstDerivativeValue(iv1d);
    }
    return feed;
}


double NNRay::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d)
{
    if (iv1d >= _vp_id_shift) {
        // if the variational parameter with index iv1d is in the ray add the following element
        return _sources[iv1d - _vp_id_shift]->getFirstDerivativeValue(i1d);
    }

    // else add source components
    double feed = 0.;
    for (size_t i = 0; i < _map_index_to_sources[iv1d].size(); ++i) {
        feed += _beta[_map_index_to_sources[iv1d][i]]*_sources[_map_index_to_sources[iv1d][i]]->getCrossFirstDerivativeValue(i1d, iv1d);
    }
    return feed;
}


double NNRay::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d)
{
    if (iv2d >= _vp_id_shift) {
        // if the variational parameter with index iv2d is in the ray add the following element
        return _sources[iv2d - _vp_id_shift]->getSecondDerivativeValue(i2d);
    }

    // else add source components
    double feed = 0.;
    for (size_t i = 0; i < _map_index_to_sources[iv2d].size(); ++i) {
        feed += _beta[_map_index_to_sources[iv2d][i]]*_sources[_map_index_to_sources[iv2d][i]]->getCrossSecondDerivativeValue(i2d, iv2d);
    }
    return feed;
}

