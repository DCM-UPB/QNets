#include "qnets/poly/feed/NNRay.hpp"

#include <cmath>
#include <random>
#include <ctime>

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

    rgen = std::mt19937_64(rdev());
    const double init_sigma = sqrt(1./_sources.size()); // beta initialization width guess
    std::normal_distribution<double> rd(0., init_sigma);

    _beta[0] = 0.; // init offset weight to 0
    for (size_t i = 1; i<_sources.size(); ++i) { _beta[i] = rd(rgen); }
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

