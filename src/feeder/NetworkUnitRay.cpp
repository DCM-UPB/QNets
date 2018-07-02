#include "NetworkUnitRay.hpp"
#include "FedNetworkUnit.hpp"
#include "StringCodeUtilities.hpp"

#include <vector>
#include <string>
#include <cmath>

// --- Feed Mu and Sigma

double NetworkUnitRay::getFeedMu()
{
    double mu = 0.;
    for (std::vector<NetworkUnit *>::size_type i=0; i<_source.size(); ++i) {
        mu += _intensity[i] * _source[i]->getOutputMu();
    }
    return mu;
}


double NetworkUnitRay::getFeedSigma()
{
    double var = 0.;
    for (std::vector<NetworkUnit *>::size_type i=0; i<_source.size(); ++i) {
        var += pow(_intensity[i] * _source[i]->getOutputSigma(), 2);
    }
    return sqrt(var);
}


// --- Betas

int NetworkUnitRay::getNBeta(){return _intensity.size();}
double NetworkUnitRay::getBeta(const int &i){return _intensity[i];}
void NetworkUnitRay::setBeta(const int &i, const double &b){_intensity[i]=b;}

// --- Variational Parameters

int NetworkUnitRay::getNVariationalParameters()
{
    return (_intensity_id_shift > -1) ? _intensity.size() : 0;
}

int NetworkUnitRay::getMaxVariationalParameterIndex()
{
    if (_intensity_id_shift > -1) {
        return _intensity_id_shift + getNVariationalParameters() - 1;
    }
    else return -1; // there are no vp in the whole feed
}

bool NetworkUnitRay::setVariationalParameterValue(const int &id, const double &value){
    if (_intensity_id_shift > -1) {
        if ( isVPIndexUsedInFeeder(id) ){
            _intensity[ id - _intensity_id_shift ] = value;
            return true;
        }
    }
    return false;
}


bool NetworkUnitRay::getVariationalParameterValue(const int &id, double &value){
    if (_intensity_id_shift > -1) {
        if ( isVPIndexUsedInFeeder(id) ){
            value = _intensity[ id - _intensity_id_shift ];
            return true;
        }
    }
    value = 0.;
    return false;
}


int NetworkUnitRay::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_betas){
    // Here we assign external vp indexes to internal indexes.
    // Also, we create two betas_used sets which are explained in the header.
    // NOTE: The current method assumes, that no index larger than max_id,
    //       max_id = starting_index + source.size() - 1 ,
    //       may be in use FOR (and trivially IN) this ray.
    // NOTE2: betas_used sets are automatically sorted -> binary_search can be used later

    _intensity_id.clear();
    _map_index_to_sources.clear();

    for (int j=0; j<starting_index; ++j) {
        std::vector<int> empty_vec;
        _map_index_to_sources.push_back(empty_vec);
    }

    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i) {
        if(FedNetworkUnit * fu = dynamic_cast<FedNetworkUnit *>(_source[i])) {
            NetworkUnitFeederInterface * feeder = fu->getFeeder();
            if (feeder != 0) {
                for (int j=0; j<starting_index; ++j) {
                    if (feeder->isVPIndexUsedForFeeder(j)) {
                        _map_index_to_sources[j].push_back(i);
                    }
                }
            }
        }
    }

    if (flag_add_betas) {
        _intensity_id_shift = starting_index;
        int idx=starting_index;
        for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i) {
            _intensity_id.push_back(idx);
            idx++;
        }

        return idx;
    }
    else {
        _intensity_id_shift = -1;
        return starting_index;
    }
}


// --- StringCode methods

std::string NetworkUnitRay::getParams()
{
    std::string id_shift_str;
    std::vector<std::string> beta_strs;

    id_shift_str = composeParamCode("id_shift", _intensity_id_shift);
    for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i) {
        beta_strs.push_back(composeParamCode("b"+std::to_string(i), _intensity[i]));
    }
    return composeCodes(id_shift_str, composeCodeList(beta_strs));
}


void NetworkUnitRay::setParams(const std::string &params)
{
    int starting_index;
    double beta;
    std::string str = readParamValue(params, "id_shift");

    if (setParamValue(str, starting_index)) this->setVariationalParametersIndexes(starting_index);

    for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i) {
        str = readParamValue(params, "b"+std::to_string(i));
        if (setParamValue(str, beta)) this->setBeta(i, beta);
    }
}


// --- Computation


double NetworkUnitRay::getFeed(){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=0; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getValue();
    }
    return feed;
}


double NetworkUnitRay::getFirstDerivativeFeed(const int &i1d){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getFirstDerivativeValue(i1d);
    }

    return feed;
}


double NetworkUnitRay::getSecondDerivativeFeed(const int &i2d){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getSecondDerivativeValue(i2d);
    }
    return feed;
}


double NetworkUnitRay::getVariationalFirstDerivativeFeed(const int &iv1d){
    double feed = 0.;

    if (iv1d < _intensity_id_shift+getNVariationalParameters()) {
        // if the variational parameter with index iv1d is in the ray add the following element
        if (iv1d >= _intensity_id_shift) {
            feed += _source[ iv1d - _intensity_id_shift ]->getValue();
        }
        else {
            // add source components
            for (size_t i=0; i<_map_index_to_sources[iv1d].size(); ++i) {
                feed += _intensity[_map_index_to_sources[iv1d][i]] * _source[_map_index_to_sources[iv1d][i]]->getVariationalFirstDerivativeValue(iv1d);
            }
        }
    }

    return feed;
}


double NetworkUnitRay::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
    double feed = 0.;

    if (iv1d < _intensity_id_shift+getNVariationalParameters()) {
        // if the variational parameter with index iv1d is in the ray add the following element
        if (iv1d >= _intensity_id_shift) {
            feed += _source[ iv1d - _intensity_id_shift ]->getFirstDerivativeValue(i1d);
        }
        else {
            // add source components
            for (size_t i=0; i<_map_index_to_sources[iv1d].size(); ++i) {
                feed += _intensity[_map_index_to_sources[iv1d][i]] * _source[_map_index_to_sources[iv1d][i]]->getCrossFirstDerivativeValue(i1d, iv1d);
            }
        }
    }

    return feed;
}


double NetworkUnitRay::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
    double feed = 0.;

    if (iv2d < _intensity_id_shift+getNVariationalParameters()) {
        // if the variational parameter with index iv2d is in the ray add the following element
        if (iv2d >= _intensity_id_shift) {
            feed += _source[ iv2d - _intensity_id_shift ]->getSecondDerivativeValue(i2d);
        }
        else {
            // add source components
            for (size_t i=0; i<_map_index_to_sources[iv2d].size(); ++i) {
                feed += _intensity[_map_index_to_sources[iv2d][i]] * _source[_map_index_to_sources[iv2d][i]]->getCrossSecondDerivativeValue(i2d, iv2d);
            }
        }
    }

    return feed;
}


// --- Beta Index

bool NetworkUnitRay::isVPIndexUsedInFeeder(const int &id){
    if ( _intensity_id_shift <= id && id <_intensity_id_shift+getNVariationalParameters()) {
        return true;
    }
    else return false;
}

bool NetworkUnitRay::isVPIndexUsedInSources(const int &id){
    if (id < _intensity_id_shift) {
        return (!_map_index_to_sources[id].empty());
    }
    else return false;
}

bool NetworkUnitRay::isVPIndexUsedForFeeder(const int &id){
    if ( isVPIndexUsedInFeeder(id) || isVPIndexUsedInSources(id) ) {
        return true;
    }
    else return false;
}


// --- Constructor

NetworkUnitRay::NetworkUnitRay(NetworkLayer * nl) {
    // target sigma to keep sum of weighted inputs in range [-4,4], assuming uniform distribution
    // sigma = 8/sqrt(12) = (b-a)/sqrt(12) * m^(1/2)
    const double bah = 4 * pow(nl->getNUnits(), -0.5); // (b-a)/2

    _rgen = std::mt19937_64(_rdev());
    _rd = std::uniform_real_distribution<double>(-bah,bah);

    for (int i=0; i<nl->getNUnits(); ++i){
        _source.push_back(nl->getUnit(i));
        _intensity.push_back(_rd(_rgen));
    }

    setVariationalParametersIndexes(nl->getMaxVariationalParameterIndex(), false); // per default we don't add betas as variational parameters
}

// --- Destructor

NetworkUnitRay::~NetworkUnitRay(){
    _source.clear();
    _intensity.clear();
    _intensity_id.clear();
    _map_index_to_sources.clear();
}
