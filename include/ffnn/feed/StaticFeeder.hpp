#ifndef STATIC_FEEDER
#define STATIC_FEEDER

#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/unit/NetworkUnit.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <string>
#include <vector>


class StaticFeeder: public FeederInterface
{
public:
    virtual ~StaticFeeder(){};

    // set string codes
    virtual void setParams(const std::string &params){
        FeederInterface::setParams(params); // we don't need more to init vp system
        if (_vp_id_shift > -1) setVariationalParametersIndexes(_vp_id_shift, false);
    }

    // --- Devirtualize methods for performance

    int getNBeta(){return 0;}
    double getBeta(const int &i){return FeederInterface::getBeta(i);}
    void setBeta(const int &i, const double &beta){FeederInterface::setBeta(i, beta);}

    // variational parameters (we don't have any, so we can default the methods)
    int getNVariationalParameters(){return 0;}
    int getMaxVariationalParameterIndex(){return FeederInterface::getMaxVariationalParameterIndex();}
    int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = false){ // we don't add vp and if there are no previous vp, we can just pretend vps aren't initialized (faster?)
        const int ret = FeederInterface::setVariationalParametersIndexes(starting_index, false); if (starting_index < 1) _vp_id_shift = -1; return ret;}
    bool getVariationalParameterValue(const int &id, double &value){return FeederInterface::getVariationalParameterValue(id, value);}
    bool setVariationalParameterValue(const int &id, const double &value){return FeederInterface::setVariationalParameterValue(id, value);}

    // IsVPIndexUsed methods
    bool isVPIndexUsedInFeeder(const int &id){return false;}  // always false
    bool isVPIndexUsedForFeeder(const int &id){return FeederInterface::isVPIndexUsedInSources(id);} // we only need to check sources

    // Randomizers
    // (we don't need any of them, since we don't have any auto-adjustable variables)
    void randomizeBeta(){}
    void randomizeParams(){}
    void randomizeVP(){}
};

#endif
