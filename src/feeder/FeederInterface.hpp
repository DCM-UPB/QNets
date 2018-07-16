#ifndef FEEDER_INTERFACE
#define FEEDER_INTERFACE

#include "SerializableComponent.hpp"

#include <string>
#include <vector>
#include <stdexcept>

class NetworkUnit;  // forward declaration to solve circular dependency
class NetworkLayer;


class FeederInterface: public SerializableComponent
{
protected:
    // sources
    std::vector<NetworkUnit *> _sourcePool; // stored pool of possible sources
    std::vector<NetworkUnit *> _sources;   // actual sources from which the feeder takes output (to be filled by child)
    std::vector<size_t> _source_ids;  // which index in sourcePool is the index in source
    std::vector<std::vector<size_t>> _map_index_to_sources; // store indices of relevant sources for each variational parameter (in sources)

    // variational parameters
    std::vector<double*> _vp; // store pointers to beta/params used as variational parameters
    int _vp_id_shift = -1; // if we add vp, our vp indices start from here (-1 means variational parameter system not initialized)
    bool _flag_vp = false; // do we add own vp?


    void _fillSourcePool(NetworkLayer * nl); // add units from nl to sourcePool
    void _fillSources(const std::vector<size_t> &source_ids); // add select sources from sourcePool
    void _fillSources(); // add all sources from sourcePool
public:
    virtual ~FeederInterface();

    // set string codes
    std::string getClassIdCode() {return "feeder";}
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // sources, i.e. the units from which the values are taken from
    int getNSources(){return _sources.size();}
    NetworkUnit * getSource(const int &i){return _sources[i];}

    // return the feed mean value (mu) and standard deviation (sigma)
    virtual double getFeedMu() = 0;
    virtual double getFeedSigma() = 0;

    // feed
    virtual double getFeed() = 0;  // e.g. get   sum_j( b_j x_j )

    // derivatives
    virtual double getFirstDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( b_j dx_j/dx0_i ), where x0 is the coordinate in the input layer
    virtual double getSecondDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( b_j d^2x_j/dx0_j^2 ), where x0 is the coordinate in the input layer

    // derivative in respect to the variational parameters
    virtual double getVariationalFirstDerivativeFeed(const int &i) = 0;  // e.g. get   sum_j( db_j/db_i x_j + b_j dx_j/db_i ), where i is the index of the variational parameter (j might not cross it)
    virtual double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d) = 0;  // e.g. get   d^2/dxdb sum_j( b_j x_j ), where i1d is the index for x, and iv1d is the index for b
    virtual double getCrossSecondDerivativeFeed(const int &i2d, const int &iv1d) = 0; // e.g. get    d^3/dx^2db sum_j( b_j x_j ), where i1d is the index for x, and iv1d is the index for b


    // beta (meaning the individual factors directly multiplied to each used source output)
    virtual int getNBeta(){return 0;}
    virtual double getBeta(const int &i){throw std::runtime_error("FeederInterface::getBeta called, but the base interface defaults to no beta. Derive from WeightedFeederInterface to use beta.");}
    virtual void setBeta(const int &i, const double &b){throw std::runtime_error("FeederInterface::setBeta called, but the base interface defaults to no beta. Derive from WeightedFeederInterface to use beta.");}

    // variational parameters
    int getNVariationalParameters();  // return the number of variational parameters involved
    int getMaxVariationalParameterIndex(); // return the highest appearing variational parameter index from the whole feed (including self). If none, return -1;
    virtual int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);  // set the index of each variational parameter starting from starting_index  and create vp pointer vector
    // return the index that the next feeder might take as input
    bool getVariationalParameterValue(const int &id, double &value); // get the variational parameter with identification index id and store it in value
    // return true if the parameters has been found, false otherwise
    bool setVariationalParameterValue(const int &id, const double &value); // set the variational parameter with identification index id with the number stored in value

    // IsVPIndexUsed methods
    // return true if the parameters has been found, false otherwise
    bool isVPIndexUsedInFeeder(const int &id);  // variational parameter is directly used?
    bool isVPIndexUsedInSources(const int &id); // variational parameter is indirectly used?
    bool isVPIndexUsedForFeeder(const int &id); // variational parameter is used directly or indirectly?

    // Randomizers
    virtual void randomizeBeta(){}; // randomize beta intensities, do nothing since we default to no betas
    virtual void randomizeParams(){} // randomize extra parameters, again do nothing by default
    virtual void randomizeVP(){}; // randomize all assigned variational parameters
};

#endif
