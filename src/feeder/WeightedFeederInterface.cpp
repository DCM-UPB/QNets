#include "WeightedFeederInterface.hpp"

// --- default fillBeta method

void WeightedFeederInterface::_fillBeta()
{
    for (size_t i=0; i<_sources.size(); ++i) _beta.push_back(0.);
    randomizeBeta();
}


// --- StringCode methods

std::string WeightedFeederInterface::getParams()
{
    std::string base_str = FeederInterface::getParams();
    std::vector<std::string> beta_strs;

    for (std::vector<double>::size_type i=0; i<_beta.size(); ++i) {
        beta_strs.push_back(composeParamCode("b"+std::to_string(i), _beta[i]));
    }
    return composeCodes(base_str, composeCodeList(beta_strs));
}

void WeightedFeederInterface::setParams(const std::string &params)
{
    FeederInterface::setParams(params);

    double beta;
    for (std::vector<double>::size_type i=0; i<_beta.size(); ++i) {
        std::string str = readParamValue(params, "b"+std::to_string(i));
        if (setParamValue(str, beta)) this->setBeta(i, beta);
    }
}


// set VP Indexes with Betas

int WeightedFeederInterface::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    int idx_base = FeederInterface::setVariationalParametersIndexes(starting_index, flag_add_vp);

    if (flag_add_vp) {
        for (double &b : _beta) {
            _vp.push_back(&b);
        }
        return _vp_id_shift + _beta.size();
    }
    else {
        return idx_base;
    }
}
