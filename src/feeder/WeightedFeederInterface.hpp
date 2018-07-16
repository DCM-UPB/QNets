#ifndef WEIGHTED_FEEDER_INTERFACE
#define WEIGHTED_FEEDER_INTERFACE

#include "FeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <string>
#include <vector>

class WeightedFeederInterface: public FeederInterface
{
protected:
    // beta
    std::vector<double> _beta;   // intensity of each sorgent unit, i.e. its weight

    // method to fill beta after source is filled
    // we provide a default method to add one beta per selected source
    virtual void _fillBeta();

public:
    virtual ~WeightedFeederInterface(){_beta.clear();}

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
