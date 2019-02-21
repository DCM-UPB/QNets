#include "ffnn/feed/WeightedFeeder.hpp"

// --- clear method

void WeightedFeeder::_clearSources()
{
    _beta.clear();
    VariableFeeder::_clearSources();
}


// --- default fillBeta method

void WeightedFeeder::_fillBeta()
{
    for (size_t i=0; i<_sources.size(); ++i) _beta.push_back(0.);
    randomizeBeta();
}


// --- StringCode methods

std::string WeightedFeeder::getParams()
{
    std::string base_str = VariableFeeder::getParams();
    std::vector<std::string> beta_strs;

    for (std::vector<double>::size_type i=0; i<_beta.size(); ++i) {
        beta_strs.push_back(composeParamCode("b"+std::to_string(i), _beta[i]));
    }
    return composeCodes(base_str, composeCodeList(beta_strs));
}

void WeightedFeeder::setParams(const std::string &params)
{
    VariableFeeder::setParams(params);

    double beta;
    for (std::vector<double>::size_type i=0; i<_beta.size(); ++i) {
        std::string str = readParamValue(params, "b"+std::to_string(i));
        if (setParamValue(str, beta)) this->setBeta(i, beta);
    }
}


// set VP Indexes with all Betas (default, override if you want something else)

int WeightedFeeder::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    int idx_base = VariableFeeder::setVariationalParametersIndexes(starting_index, flag_add_vp);

    if (_flag_vp) {
        for (double &b : _beta) {
            _vp.push_back(&b);
        }
        return _vp_id_shift + _beta.size();
    }
    else {
        return idx_base;
    }
}
