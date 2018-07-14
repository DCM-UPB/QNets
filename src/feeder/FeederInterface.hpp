#ifndef FEEDER_INTERFACE
#define FEEDER_INTERFACE

#include "SerializableComponent.hpp"

#include <string>
#include <vector>

class NetworkUnit;  // forward declaration to solve circular dependency
class NetworkLayer;

class FeederInterface: public SerializableComponent
{
protected:
    // sources
    std::vector<NetworkUnit *> _sourcePool; // stored pool of possible sources
    std::vector<NetworkUnit *> _source;   // actual sources from which the feeder takes output (to be filled by child)
    std::vector<int> _source_ids;  // which index in sourcePool is the index in source
    std::vector<std::vector<int>> _map_index_to_sources; // store indices of relevant sources for each variational parameter (in sources)

    // beta / variational parameters
    std::vector<double> _beta;   // intensity of each sorgent unit, i.e. its weight

    int _vp_id_shift; // if we add vp, our vp indices start from here (-1 means variational parameter system not initialized)

public:
    explicit FeederInterface(NetworkLayer * nl);
    virtual ~FeederInterface();

    // set string codes
    std::string getClassIdCode() {return "feeder";}
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // return the feed mean value (mu) and standard deviation (sigma)
    virtual double getFeedMu() = 0;
    virtual double getFeedSigma() = 0;

    // sources, i.e. the units from which the values are taken from
    int getNSources(){return _source.size();}
    NetworkUnit * getSource(const int &i){return _source[i];}

    // feed
    virtual double getFeed() = 0;  // e.g. get   sum_j( b_j x_j )

    // derivatives
    virtual double getFirstDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( b_j dx_j/dx0_i ), where x0 is the coordinate in the input layer
    virtual double getSecondDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( b_j d^2x_j/dx0_j^2 ), where x0 is the coordinate in the input layer

    // beta (meaning the individual factors directly multiplied to each used source output)
    int getNBeta(){return _beta.size();}
    double getBeta(const int &i){return _beta[i];}
    void setBeta(const int &i, const double &b){_beta[i]=b;}

    // variational parameters
    virtual int getNVariationalParameters(){return 0;}  // return the number of variational parameters involved
    virtual int getMaxVariationalParameterIndex(){return _vp_id_shift;} // return the highest appearing variational parameter index from the whole feed (including self). If none, return -1;
    virtual int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);  // set the identification index of each variational parameter starting from the given input value
    // return the index that the next feeder might take as input
    virtual bool getVariationalParameterValue(const int &id, double &value){return false;} // get the variational parameter with identification index id and store it in value
    // return true if the parameters has been found, false otherwise
    virtual bool setVariationalParameterValue(const int &id, const double &value){return false;} // set the variational parameter with identification index id with the number stored in value
    // return true if the parameters has been found, false otherwise
    // derivative in respect to the variational parameters
    virtual double getVariationalFirstDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( db_j/db_i x_j + b_j dx_j/db_i ), where i is the index of the variational parameter (j might not cross it)
    virtual double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d) = 0;  // e.g. get   d^2/dxdb sum_j( b_j x_j ), where i1d is the index for x, and iv1d is the index for b
    virtual double getCrossSecondDerivativeFeed(const int &i2d, const int &iv1d) = 0; // e.g. get    d^3/dx^2db sum_j( b_j x_j ), where i1d is the index for x, and iv1d is the index for b

    bool isVPIndexUsedInFeeder(const int &id);  // variational parameter is directly used?
    bool isVPIndexUsedInSources(const int &id); // variational parameter is indirectly used?
    bool isVPIndexUsedForFeeder(const int &id); // variational parameter is used directly or indirectly?
};

#endif
