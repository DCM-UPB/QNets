#ifndef NETWORK_UNIT_RAY
#define NETWORK_UNIT_RAY

#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <vector>
#include <random>
#include <set>
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

    // store information about which beta are used in or for this ray
    // NB: 'in' this ray means that the beta is part of the ray
    //     'for' this ray means that the beta is either used in this ray or in another
    //           ray that genreates an output that is directly or indirectly used
    //           in this ray (sources)
    std::set<int> _betas_used_in_this_ray;
    std::set<int> _betas_used_for_this_ray;

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
    bool isVPIndexUsedInThisRay(const int &id);   // variational parameter is directly used?
    bool isVPIndexUsedForThisRay(const int &id);   // variational parameter is used directly or indirectly?
};

#endif
