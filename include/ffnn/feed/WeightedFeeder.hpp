#ifndef WEIGHTED_FEEDER
#define WEIGHTED_FEEDER

#include "ffnn/feed/VariableFeeder.hpp"
#include "ffnn/unit/NetworkUnit.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <string>
#include <vector>

class WeightedFeeder: public VariableFeeder
{
protected:
    // beta
    std::vector<double> _beta;   // intensity of each sorgent unit, i.e. its weight

    virtual void _clearSources(); // basically clear everything except sourcePool

    // method to fill beta after source is filled
    // we provide a method to add one beta per selected source
    void _fillBeta();

public:
    virtual ~WeightedFeeder(){_beta.clear();}

    // set string codes
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // beta (meaning the individual factors directly multiplied to each used source output)
    int getNBeta(){return _beta.size();}
    double getBeta(const int &i){return _beta[i];}
    void setBeta(const int &i, const double &b){_beta[i]=b;}

    // provide default setVPIndexes for the case that all beta are added as vp
    virtual int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);

    // randomizers
    // we provide a default vp randomizer for the case that all beta are added as vp
    virtual void randomizeVP(){if (_flag_vp) randomizeBeta();}
};

#endif
