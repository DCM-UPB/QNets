#include "ffnn/feed/VariableFeeder.hpp"

#include <string>

// --- clear method

void VariableFeeder::_clearSources()
{
    _vp.clear();
    FeederInterface::_clearSources();
}

// --- StringCode methods

std::string VariableFeeder::getParams()
{
    return composeCodes(FeederInterface::getParams(), composeParamCode("flag_vp", _flag_vp));
}


void VariableFeeder::setParams(const std::string &params)
{
    FeederInterface::setParams(params);
    std::string str_vp = readParamValue(params, "flag_vp");
    setParamValue(str_vp, _flag_vp);
    // in the child classes versions you need to do something like this:
    // if (_vp_id_shift > -1) this->setVariationalParametersIndexes(_vp_id_shift, _flag_vp);
}

// set VP Indexes default version

int VariableFeeder::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    // NOTE: Extend this to actually add variational parameters

    FeederInterface::setVariationalParametersIndexes(starting_index, flag_add_vp);
    _vp.clear();
    _flag_vp = flag_add_vp;

    return starting_index;
}


// --- Variational Parameters

int VariableFeeder::getNVariationalParameters()
{
    return _vp.size();
}

int VariableFeeder::getMaxVariationalParameterIndex()
{
    if (_vp_id_shift > -1) {
        if(_flag_vp) {
            return _vp_id_shift + _vp.size() - 1;
        }
        else return _vp_id_shift;
    }
    else return -1; // vp not initialized
}

bool VariableFeeder::setVariationalParameterValue(const int &id, const double &value){
    if (_flag_vp) {
        if ( isVPIndexUsedInFeeder(id) ){
            *_vp[ id - _vp_id_shift ] = value;
            return true;
        }
    }
    return false;
}


bool VariableFeeder::getVariationalParameterValue(const int &id, double &value){
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

bool VariableFeeder::isVPIndexUsedInFeeder(const int &id)
{
    if ( _vp_id_shift <= id && id <_vp_id_shift+(int)_vp.size()) {
        return true;
    }
    else return false;
}

