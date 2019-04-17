#include "qnets/feed/SmartBetaGenerator.hpp"

#include <numeric>
#include <random>


namespace smart_beta
{

namespace details
{
// little internal cast helper
NNRay * _castFeederToRay(FeederInterface * const feeder)
{
    if (auto * ray = dynamic_cast<NNRay *>(feeder)) {
        return ray;
    }

    return nullptr;
}

std::vector<int> _findIndexesOfUnitsWithRay(FedLayer * L)
{
    //
    // Given a fed layer, find the indexes of the units which have a ray
    //
    using namespace std;

    vector<int> idx;
    for (int i = 0; i < L->getNFedUnits(); ++i) {
        if (_castFeederToRay(L->getFedUnit(i)->getFeeder()) != nullptr) {
            idx.push_back(i);
        }
    }

    return idx;
}


void _computeBetaMuAndSigma(FedUnit * U, double &mu, double &sigma)
{
    //
    // Use the formulas:
    //   mu_beta = mu_actf_input / (sum_U_sources mu_actf_output)
    //   sigma_beta = sigma_actf_input / (sum_U_sources mu_actf_output)
    //
    FeederInterface * ray = U->getFeeder();
    if (ray == nullptr) {
        throw std::runtime_error("Provided unit does not have a ray, therefore it does not need beta");
    }

    // first compute the source mu / sigma
    double mu_source = 0;
    double sigma_source = 0;

    for (int i = BETA_INDEX_OFFSET; i < ray->getNSources(); ++i) { // leave out offset sources here, assuming first source is always offset!
        mu_source += ray->getSource(i)->getOutputMu();
        sigma_source += ray->getSource(i)->getOutputSigma();
    }

    // assign the correct values
    mu = (mu_source != 0) ? U->getIdealProtoMu()/mu_source : 0.;
    sigma = (sigma_source != 0) ? U->getIdealProtoSigma()/sigma_source : 1.;
}


void _setRandomBeta(FeederInterface * ray, const double &mu, const double &sigma)
{
    //
    // Sample the beta of a given ray using a normal distribution with mean mu and standard deviation sigma.
    //
    using namespace std;

    random_device rdev;
    mt19937_64 rgen = mt19937_64(rdev());
    normal_distribution<double> norm(mu, sigma);

    for (int i = BETA_INDEX_OFFSET; i < ray->getNBeta(); ++i) {
        ray->setBeta(i, norm(rgen));
    }
}


void _makeBetaOrthogonal(FeederInterface * fixed_ray, FeederInterface * ray)
{
    //
    // Modify ray in order to make it orthogonal to fixed_ray, preserving the norm
    //
    using namespace std;

    // build vectors v and u, corresponding to the betas of the two rays
    vector<double> v;
    for (int i = BETA_INDEX_OFFSET; i < fixed_ray->getNBeta(); ++i) {
        v.push_back(fixed_ray->getBeta(i));
    }
    vector<double> u;
    for (int i = BETA_INDEX_OFFSET; i < ray->getNBeta(); ++i) {
        u.push_back(ray->getBeta(i));
    }

    // make u orthogonal to v
    const double u_u = inner_product(begin(u), end(u), begin(u), 0.0);
    const double u_v = inner_product(begin(u), end(u), begin(v), 0.0);
    const double v_v = inner_product(begin(v), end(v), begin(v), 0.0);
    const double orthog_fact = u_v/v_v;
    for (unsigned int i = 0; i < u.size(); ++i) {
        u[i] -= orthog_fact*v[i];
    }

    // compute the new norm of u
    const double new_u_u = inner_product(begin(u), end(u), begin(u), 0.0);
    if (sqrt(new_u_u) < MIN_BETA_NORM) {
        throw std::runtime_error("By applying the orthogonality procedure, the norm of the beta vector has been reduced too much");
    }

    // set the values of u to the ray
    for (int i = BETA_INDEX_OFFSET; i < ray->getNBeta(); ++i) {
        ray->setBeta(i, u[i - BETA_INDEX_OFFSET]*sqrt(u_u/new_u_u));
    }
}
}  // namespace details

// --- External methods in smart_beta namespace

void generateSmartBeta(FeedForwardNeuralNetwork * ffnn)
{
    for (int i = 0; i < ffnn->getNFedLayers(); ++i) {
        generateSmartBeta(ffnn->getFedLayer(i));
    }
}


void generateSmartBeta(FedLayer * L)
{
    using namespace std;
    using namespace details;

    // get the indexes of the units we are interested in
    vector<int> idx = _findIndexesOfUnitsWithRay(L);

    if (!idx.empty()) {
        // set the first unit beta
        FedUnit * u0 = L->getFedUnit(idx[0]);
        double mu, sigma;
        _computeBetaMuAndSigma(u0, mu, sigma);
        FeederInterface * ray0 = u0->getFeeder();
        _setRandomBeta(ray0, mu, sigma);

        // set the other unit beta until a complete basis set is formed
        const int n_li_units = (signed(idx.size()) <= ray0->getNBeta() - BETA_INDEX_OFFSET)
                               ? signed(idx.size())
                               : ray0->getNBeta() - BETA_INDEX_OFFSET;
        for (int i = 1; i < n_li_units; ++i) {
            FedUnit * u = L->getFedUnit(idx[i]);
            FeederInterface * ray = u->getFeeder();
            double mu, sigma;
            bool flag_sample_random_beta = true;
            while (flag_sample_random_beta) {
                _computeBetaMuAndSigma(u, mu, sigma);
                _setRandomBeta(ray, mu, sigma);
                flag_sample_random_beta = false;
                for (int j = 0; j < i; ++j) {
                    try {
                        _makeBetaOrthogonal(L->getFedUnit(idx[j])->getFeeder(), ray);
                    }
                    catch (std::runtime_error &e) {
                        // by applying the orthogonality prcedure, the norm of the beta vector has been reduced too much...
                        // This means the vector had the same direction of a previous one and it was not possible to generate a orthogonal one
                        flag_sample_random_beta = true;
                    }
                }
            }
        }

        // set all the remaninig unit beta, which will be redundant / linear dependent
        for (int i = n_li_units; i < signed(idx.size()); ++i) {
            FedUnit * u = L->getFedUnit(idx[i]);
            FeederInterface * ray = u->getFeeder();
            double mu, sigma;
            _computeBetaMuAndSigma(u, mu, sigma);
            double best_dot_product = -1.;
            vector<double> best_beta;
            for (int ib = BETA_INDEX_OFFSET; ib < ray->getNBeta(); ++ib) {
                best_beta.push_back(0.);
            }
            for (int itry = 0; itry < N_TRY_BEST_LD_BETA; ++itry) {
                _setRandomBeta(ray, mu, sigma);
                vector<double> beta_u;
                for (int ib = BETA_INDEX_OFFSET; ib < ray->getNBeta(); ++ib) {
                    beta_u.push_back(ray->getBeta(ib));
                }
                double min_dot_product = -1.;
                for (int j = 0; j < i; ++j) {
                    vector<double> beta_v;
                    FeederInterface * rayj = L->getFedUnit(idx[j])->getFeeder();
                    for (int ib = BETA_INDEX_OFFSET; ib < rayj->getNBeta(); ++ib) {
                        beta_v.push_back(rayj->getBeta(ib));
                    }
                    const double dot_product = fabs(inner_product(begin(beta_u), end(beta_u), begin(beta_v), 0.0))/inner_product(begin(beta_u), end(beta_u), begin(beta_u), 0.0);
                    if (min_dot_product < 0.) {
                        min_dot_product = dot_product;
                    }
                    if (dot_product < min_dot_product) {
                        min_dot_product = dot_product;
                    }
                }
                if (best_dot_product < 0.) {
                    best_dot_product = min_dot_product;
                }
                if (min_dot_product < best_dot_product) {
                    best_dot_product = min_dot_product;
                    for (int ib = BETA_INDEX_OFFSET; ib < ray->getNBeta(); ++ib) {
                        best_beta[ib - BETA_INDEX_OFFSET] = beta_u[ib - BETA_INDEX_OFFSET];
                    }
                }
            }
            for (int ib = BETA_INDEX_OFFSET; ib < ray->getNBeta(); ++ib) {
                ray->setBeta(ib, best_beta[ib - BETA_INDEX_OFFSET]);
            }
        }
    }
}
} // namespace smart_beta
