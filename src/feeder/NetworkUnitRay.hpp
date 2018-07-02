#ifndef NETWORK_UNIT_RAY
#define NETWORK_UNIT_RAY

#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <vector>
#include <random>
#include <map>
#include <string>

class NetworkUnitRay: public NetworkUnitFeederInterface {
protected:
    // random number generator, used to initialize the intensities
    std::random_device _rdev;
    std::mt19937_64 _rgen;
    std::uniform_real_distribution<double> _rd;

    // key component of the ray: the source and their intensisities
    std::vector<NetworkUnit *> _source;   // units from which the ray takes the values from
    std::vector<double> _intensity;   // intensity of each sorgent unit, i.e. its weight
    std::vector<int> _intensity_id;  // intensity identification id, useful for the NN
    int _intensity_id_shift;  // shift of the previous vector

    // store indices of relevant sources for each variational parameter (in sources)
    std::vector<std::vector<int>> _map_index_to_sources;
public:
    explicit NetworkUnitRay(NetworkLayer * nl);
    ~NetworkUnitRay();

    // string code methods
    std::string getIdCode(){return "RAY";}; // return an identification string
    std::string getParams();
    void setParams(const std::string &params);

    // return the feed mean value (mu) and standard deviation (sigma)
    double getFeedMu();
    double getFeedSigma();

    // sources
    int getNSources(){return _source.size();}
    NetworkUnit * getSource(const int &i){return _source[i];}

    // beta
    int getNBeta();
    double getBeta(const int &i);
    void setBeta(const int &i, const double &b);

    // Variational Parameters
    int getNVariationalParameters();
    int getMaxVariationalParameterIndex();
    int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_betas = true);
    bool getVariationalParameterValue(const int &id, double &value);
    bool setVariationalParameterValue(const int &id, const double &value);

    // Computation
    double getFeed();
    double getFirstDerivativeFeed(const int &i1d);
    double getSecondDerivativeFeed(const int &i2d);
    double getVariationalFirstDerivativeFeed(const int &iv1d);
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d);
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d);

    // Beta Index
    bool isVPIndexUsedInFeeder(const int &id);
    bool isVPIndexUsedInSources(const int &id);
    bool isVPIndexUsedForFeeder(const int &id);
};

#endif
