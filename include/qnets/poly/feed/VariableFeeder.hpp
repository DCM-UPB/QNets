#ifndef FFNN_FEED_VARIABLEFEEDER_HPP
#define FFNN_FEED_VARIABLEFEEDER_HPP

#include "qnets/poly/feed/FeederInterface.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"
#include "qnets/poly/unit/NetworkUnit.hpp"

#include <string>
#include <vector>


class VariableFeeder: public FeederInterface
{
protected:
    // variational parameters
    std::vector<double *> _vp; // store pointers to beta/params used as variational parameters
    bool _flag_vp = false; // do we add own vp?

    void _clearSources() override; // basically clear everything except sourcePool

public:
    ~VariableFeeder() override { _vp.clear(); }

    // set string codes
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // variational parameters
    int getNVariationalParameters() override;
    int getMaxVariationalParameterIndex() override;
    int setVariationalParametersIndexes(const int &starting_index, bool flag_add_vp = true) override;
    bool getVariationalParameterValue(const int &id, double &value) override;
    bool setVariationalParameterValue(const int &id, const double &value) override;

    // final IsVPIndexUsed methods
    bool isVPIndexUsedInFeeder(const int &id) override;
    bool isVPIndexUsedForFeeder(const int &id) override { return FeederInterface::isVPIndexUsedForFeeder(id); }
};

#endif
