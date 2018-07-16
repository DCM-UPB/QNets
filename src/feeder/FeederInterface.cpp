#include "FeederInterface.hpp"
#include "NetworkLayer.hpp"
#include "NetworkUnit.hpp"
#include "FedUnit.hpp"
#include <vector>

// --- Base Desctructor

FeederInterface::~FeederInterface()
{
    _sourcePool.clear();
    _sources.clear();
    _source_ids.clear();
    _map_index_to_sources.clear();
    _vp.clear();
}


// --- fillSources methods

void FeederInterface::_fillSourcePool(NetworkLayer * nl)
{
    for (int i=0; i<nl->getNUnits(); ++i){
        _sourcePool.push_back(nl->getUnit(i));
    }
}

void FeederInterface::_fillSources(const std::vector<size_t> &source_ids) // add select sources from sourcePool
{
    for (size_t i=0; i<source_ids.size(); ++i) {
        _source_ids.push_back(source_ids[i]);
        _sources.push_back(_sourcePool[_source_ids[i]]);
    }
}

void FeederInterface::_fillSources() // add all sources from sourcePool
{
    for (size_t i=0; i<_sourcePool.size(); ++i) {
        _source_ids.push_back(i);
        _sources.push_back(_sourcePool[i]);
    }
}


// --- StringCode methods

std::string FeederInterface::getParams()
{
    return composeCodes(composeParamCode("id_shift", _vp_id_shift), composeParamCode("flag_vp", _flag_vp));
}


void FeederInterface::setParams(const std::string &params)
{
    int starting_index;
    bool flag_vp;
    std::string str_id = readParamValue(params, "id_shift");
    std::string str_vp = readParamValue(params, "flag_vp");
    if (setParamValue(str_id, starting_index) && setParamValue(str_vp, flag_vp)) this->setVariationalParametersIndexes(starting_index, flag_vp);
}

// set VP Indexes default version

int FeederInterface::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    // Here we assign external vp indexes to internal indexes.
    // NOTE: The current method assumes, that no index relevant
    // to this feeder is larger than starting_index (+ n_own_vp - 1)
    // NOTE 2: Extend this to actually add variational parameters

    _map_index_to_sources.clear();
    _vp.clear();

    for (int j=0; j<starting_index; ++j) {
        std::vector<size_t> empty_vec;
        _map_index_to_sources.push_back(empty_vec);
    }

    for (std::vector<NetworkUnit *>::size_type i=1; i<_sources.size(); ++i) {
        if(FedUnit * fu = dynamic_cast<FedUnit *>(_sources[i])) {
            FeederInterface * feeder = fu->getFeeder();
            if (feeder != 0) {
                for (int j=0; j<starting_index; ++j) {
                    if (feeder->isVPIndexUsedForFeeder(j)) {
                        _map_index_to_sources[j].push_back(i);
                    }
                }
            }
        }
    }

    _flag_vp = flag_add_vp;
    _vp_id_shift = starting_index;
    return starting_index;
}


// --- Variational Parameters

int FeederInterface::getNVariationalParameters()
{
    return _vp.size();
}

int FeederInterface::getMaxVariationalParameterIndex()
{
    if (_vp_id_shift > -1) {
        if(_flag_vp) {
            return _vp_id_shift + _vp.size() - 1;
        }
        else return _vp_id_shift;
    }
    else return -1; // vp not initialized
}

bool FeederInterface::setVariationalParameterValue(const int &id, const double &value){
    if (_flag_vp) {
        if ( isVPIndexUsedInFeeder(id) ){
            *_vp[ id - _vp_id_shift ] = value;
            return true;
        }
    }
    return false;
}


bool FeederInterface::getVariationalParameterValue(const int &id, double &value){
    if (_flag_vp) {
        if ( isVPIndexUsedInFeeder(id) ){
            value = *_vp[ id - _vp_id_shift ];
            return true;
        }
    }
    value = 0.;
    return false;
}


// --- is VP Index used

bool FeederInterface::isVPIndexUsedInFeeder(const int &id)
{
    if ( _vp_id_shift <= id && id <_vp_id_shift+(int)_vp.size()) {
        return true;
    }
    else return false;
}

bool FeederInterface::isVPIndexUsedInSources(const int &id)
{
    if (id < _vp_id_shift) {
        return (!_map_index_to_sources[id].empty());
    }
    else return false;
}

bool FeederInterface::isVPIndexUsedForFeeder(const int &id)
{
    if ( isVPIndexUsedInFeeder(id) || isVPIndexUsedInSources(id) ) {
        return true;
    }
    else return false;
}
