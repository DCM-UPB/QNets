#ifndef VARIABLE_FEEDER
#define VARIABLE_FEEDER

#include "FeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <string>
#include <vector>


class VariableFeeder: public FeederInterface
{
protected:
    // variational parameters
    std::vector<double*> _vp; // store pointers to beta/params used as variational parameters
    bool _flag_vp = false; // do we add own vp?

    virtual void _clearSources(); // basically clear everything except sourcePool

public:
    virtual ~VariableFeeder(){_vp.clear();}

    // set string codes
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // variational parameters
    int getNVariationalParameters();
    int getMaxVariationalParameterIndex();
    virtual int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);
    bool getVariationalParameterValue(const int &id, double &value);
    bool setVariationalParameterValue(const int &id, const double &value);

    // final IsVPIndexUsed methods
    bool isVPIndexUsedInFeeder(const int &id);
    bool isVPIndexUsedForFeeder(const int &id){return FeederInterface::isVPIndexUsedForFeeder(id);}
};

#endif
