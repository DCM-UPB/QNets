#ifndef FFNN_FEED_STATICFEEDER_HPP
#define FFNN_FEED_STATICFEEDER_HPP

#include "qnets/feed/FeederInterface.hpp"
#include "qnets/layer/NetworkLayer.hpp"
#include "qnets/unit/NetworkUnit.hpp"

#include <string>
#include <vector>


class StaticFeeder: public FeederInterface
{
public:
    ~StaticFeeder() override = default;;

    // set string codes
    void setParams(const std::string &params) override
    {
        FeederInterface::setParams(params); // we don't need more to init vp system
        if (_vp_id_shift > -1) {
            setVariationalParametersIndexes(_vp_id_shift, false);
        }
    }

    // --- Devirtualize methods for performance

    int getNBeta() override { return 0; }
    double getBeta(const int &i) override { return FeederInterface::getBeta(i); }
    void setBeta(const int &i, const double &beta) override { FeederInterface::setBeta(i, beta); }

    // variational parameters (we don't have any, so we can default the methods)
    int getNVariationalParameters() override { return 0; }
    int getMaxVariationalParameterIndex() override { return FeederInterface::getMaxVariationalParameterIndex(); }
    int setVariationalParametersIndexes(const int &starting_index, const bool  /*flag_add_vp*/ = false) override
    { // we don't add vp and if there are no previous vp, we can just pretend vps aren't initialized (faster?)
        const int ret = FeederInterface::setVariationalParametersIndexes(starting_index, false);
        if (starting_index < 1) {
            _vp_id_shift = -1;
        }
        return ret;
    }
    bool getVariationalParameterValue(const int &id, double &value) override { return FeederInterface::getVariationalParameterValue(id, value); }
    bool setVariationalParameterValue(const int &id, const double &value) override { return FeederInterface::setVariationalParameterValue(id, value); }

    // IsVPIndexUsed methods
    bool isVPIndexUsedInFeeder(const int & /*id*/) override { return false; }  // always false
    bool isVPIndexUsedForFeeder(const int &id) override { return FeederInterface::isVPIndexUsedInSources(id); } // we only need to check sources

    // Randomizers
    // (we don't need any of them, since we don't have any auto-adjustable variables)
    void randomizeBeta() override {}
    void randomizeParams() override {}
    void randomizeVP() override {}
};

#endif
