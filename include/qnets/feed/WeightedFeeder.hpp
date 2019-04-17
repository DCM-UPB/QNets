#ifndef FFNN_FEED_WEIGHTEDFEEDER_HPP
#define FFNN_FEED_WEIGHTEDFEEDER_HPP

#include "qnets/feed/VariableFeeder.hpp"
#include "qnets/layer/NetworkLayer.hpp"
#include "qnets/unit/NetworkUnit.hpp"

#include <string>
#include <vector>

class WeightedFeeder: public VariableFeeder
{
protected:
    // beta
    std::vector<double> _beta;   // intensity of each sorgent unit, i.e. its weight

    void _clearSources() override; // basically clear everything except sourcePool

    // method to fill beta after source is filled
    // we provide a method to add one beta per selected source
    void _fillBeta();

public:
    ~WeightedFeeder() override { _beta.clear(); }

    // set string codes
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // beta (meaning the individual factors directly multiplied to each used source output)
    int getNBeta() override { return _beta.size(); }
    double getBeta(const int &i) override { return _beta[i]; }
    void setBeta(const int &i, const double &b) override { _beta[i] = b; }

    // provide default setVPIndexes for the case that all beta are added as vp
    int setVariationalParametersIndexes(const int &starting_index, bool flag_add_vp = true) override;

    // randomizers
    // we provide a default vp randomizer for the case that all beta are added as vp
    void randomizeVP() override
    {
        if (_flag_vp) {
            randomizeBeta();
        }
    }
};

#endif
