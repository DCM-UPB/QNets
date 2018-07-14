#ifndef NN_RAY
#define NN_RAY

#include "FeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <vector>
#include <random>
#include <map>
#include <string>

class NNRay: public FeederInterface
{
protected:
    // random number generator, used to initialize the intensities
    std::random_device _rdev;
    std::mt19937_64 _rgen;
    std::uniform_real_distribution<double> _rd;

public:
    explicit NNRay(NetworkLayer * nl);
    ~NNRay(){};

    // string code methods
    std::string getIdCode(){return "RAY";}; // return an identification string
    std::string getParams();
    void setParams(const std::string &params);

    // return the feed mean value (mu) and standard deviation (sigma)
    double getFeedMu();
    double getFeedSigma();

    // Variational Parameters
    int getNVariationalParameters();
    int getMaxVariationalParameterIndex();
    int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);
    bool getVariationalParameterValue(const int &id, double &value);
    bool setVariationalParameterValue(const int &id, const double &value);

    // Computation
    double getFeed();
    double getFirstDerivativeFeed(const int &i1d);
    double getSecondDerivativeFeed(const int &i2d);
    double getVariationalFirstDerivativeFeed(const int &iv1d);
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d);
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d);
};

#endif
