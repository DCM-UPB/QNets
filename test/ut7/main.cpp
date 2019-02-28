#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
#include <vector>

#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/feed/SmartBetaGenerator.hpp"
#include "ffnn/io/PrintUtilities.hpp"
#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "ffnn/unit/FedUnit.hpp"


int main(){
    using namespace std;

    double mu, sigma;
    bool throw_exception;
    int BETA_INDEX_OFFSET = smart_beta::details::BETA_INDEX_OFFSET;

    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);

    // --- try without feeders
    try {
        smart_beta::details::_computeBetaMuAndSigma(ffnn->getFedLayer(0)->getFedUnit(0), mu, sigma);
        throw_exception = false;
    } catch (exception &e) {
        throw_exception = true;
    }
    assert(throw_exception);


    // --- connect
    ffnn->connectFFNN();
    ffnn->getNNLayer(0)->getNNUnit(3)->setActivationFunction(std_actf::provideActivationFunction("GSS"));
    ffnn->getNNLayer(1)->getNNUnit(0)->setActivationFunction(std_actf::provideActivationFunction("GSS"));


    // --- _findIndexesOfUnitsWithRay
    vector<int> idx;
    idx = smart_beta::details::_findIndexesOfUnitsWithRay(ffnn->getFedLayer(0));
    assert( idx.size() == 4);
    assert( idx[0] == 0 );
    assert( idx[1] == 1 );
    assert( idx[2] == 2 );
    assert( idx[3] == 3 );
    idx = smart_beta::details::_findIndexesOfUnitsWithRay(ffnn->getFedLayer(1));
    assert( idx.size() == 2);
    assert( idx[0] == 0 );
    assert( idx[1] == 1 );


    // --- _computeBetaMuAndSigma

    // 4 units
    for (int il=0; il<ffnn->getFedLayer(0)->getNFedUnits(); ++il){
        smart_beta::details::_computeBetaMuAndSigma(ffnn->getFedLayer(0)->getFedUnit(il), mu, sigma);
        double mu_source = 0.;
        for (int j=BETA_INDEX_OFFSET; j<ffnn->getLayer(0)->getNUnits(); ++j){
            mu_source += ffnn->getLayer(0)->getUnit(j)->getOutputMu();
        }
        const double mu_check = (mu_source != 0) ? ffnn->getFedLayer(0)->getFedUnit(il)->getIdealProtoMu() / mu_source : 0;
        double sigma_source = 0.;
        for (int j=BETA_INDEX_OFFSET; j<ffnn->getLayer(0)->getNUnits(); ++j){
            sigma_source += ffnn->getLayer(0)->getUnit(j)->getOutputSigma();
        }
        const double sigma_check = (sigma_source != 0) ? ffnn->getFedLayer(0)->getFedUnit(il)->getIdealProtoSigma() / sigma_source : 1;
        assert( mu == mu_check );
        assert( sigma == sigma_check );
    }

    // output layer
    // 2 output units
    for (int il=0; il<ffnn->getFedLayer(1)->getNFedUnits(); ++il){
        smart_beta::details::_computeBetaMuAndSigma(ffnn->getFedLayer(1)->getFedUnit(il), mu, sigma);
        double mu_source = 0.;
        for (int j=BETA_INDEX_OFFSET; j<ffnn->getLayer(1)->getNUnits(); ++j){
            mu_source += ffnn->getLayer(1)->getUnit(j)->getOutputMu();
        }
        const double mu_check = (mu_source != 0) ? ffnn->getFedLayer(1)->getFedUnit(il)->getIdealProtoMu() / mu_source : 0;
        double sigma_source = 0.;
        for (int j=BETA_INDEX_OFFSET; j<ffnn->getLayer(1)->getNUnits(); ++j){
            sigma_source += ffnn->getLayer(1)->getUnit(j)->getOutputSigma();
        }
        const double sigma_check = (sigma_source != 0) ? ffnn->getFedLayer(1)->getFedUnit(il)->getIdealProtoSigma() / sigma_source : 1;
        assert( mu == mu_check );
        assert( sigma == sigma_check );
    }


    // --- _setRandomBeta
    FedUnit * u = ffnn->getFedLayer(0)->getFedUnit(0);
    FeederInterface * feeder = u->getFeeder();
    // compute mu and beta
    smart_beta::details::_computeBetaMuAndSigma(u, mu, sigma);
    // sample N times the beta
    const int N = 10000;
    vector<double> betas;
    for (int i=0; i<N; ++i){
        smart_beta::details::_setRandomBeta(feeder, mu, sigma);
        for (int ib=BETA_INDEX_OFFSET; ib<feeder->getNBeta(); ++ib){
            betas.push_back(feeder->getBeta(ib));
        }
    }
    // compute the mean value of the betas and check that is equal to mu
    double mean = 0.;
    for (double b : betas){
        mean += b;
    }
    mean /= (N*(feeder->getNBeta()-BETA_INDEX_OFFSET));
    // compute the standard deviation of the betas and check that is equal to mu
    double std_dev = 0.;
    for (double b : betas){
        std_dev += pow(b-mean, 2);
    }
    std_dev = sqrt(std_dev/(N*(feeder->getNBeta()-BETA_INDEX_OFFSET)-1));
    // assertions
    assert( abs(mu-mean) < 0.1 );
    assert( abs(sigma-std_dev) < 0.05 );


    // --- _makeBetaOrthogonal
    // set random beta for a unit that will stay fixed
    FedUnit * fixed_u = ffnn->getFedLayer(0)->getFedUnit(1);
    FeederInterface * fixed_feeder = fixed_u->getFeeder();
    smart_beta::details::_computeBetaMuAndSigma(fixed_u, mu, sigma);
    smart_beta::details::_setRandomBeta(fixed_feeder, mu, sigma);
    // set random beta for a unit that will be made orthogonal to the previously defined unit
    smart_beta::details::_computeBetaMuAndSigma(u, mu, sigma);
    smart_beta::details::_setRandomBeta(feeder, mu, sigma);
    // apply the orthogonaliy procedure
    smart_beta::details::_makeBetaOrthogonal(fixed_feeder, feeder);
    // compute the dot product and check that it is almost zero
    double dot_product = 0.;
    for (int i=BETA_INDEX_OFFSET; i<feeder->getNBeta(); ++i){
        dot_product += feeder->getBeta(i) * fixed_feeder->getBeta(i);
    }
    //cout << "_makeBetaOrthogonal (pair of feeders): " << dot_product << endl;
    assert( abs(dot_product) < 0.000000001);

    // now check that the norm is preserved by applying the orthogonality procedure

    // compute mu and beta
    smart_beta::details::_computeBetaMuAndSigma(u, mu, sigma);
    // sample N times the beta
    betas.clear();
    for (int i=0; i<N; ++i){
        smart_beta::details::_setRandomBeta(feeder, mu, sigma);
        smart_beta::details::_makeBetaOrthogonal(fixed_feeder, feeder);
        for (int ib=BETA_INDEX_OFFSET; ib<feeder->getNBeta(); ++ib){
            betas.push_back(feeder->getBeta(ib));
        }
    }
    // compute the mean value of the betas and check that is equal to mu
    mean = 0.;
    for (double b : betas){
        mean += b;
    }
    mean /= (N*(feeder->getNBeta()-BETA_INDEX_OFFSET));
    // compute the standard deviation of the betas and check that is equal to mu
    std_dev = 0.;
    for (double b : betas){
        std_dev += pow(b-mean, 2);
    }
    std_dev = sqrt(std_dev/(N*(feeder->getNBeta()-BETA_INDEX_OFFSET)-1));
    // assertions
    assert( abs(mu-mean) < 0.1 );
    assert( abs(sigma-std_dev) < 0.05 );


    // --- generateSmartBeta
    // generate smart beta for the whole FFNN
    smart_beta::generateSmartBeta(ffnn);
    // check the orthogonality relations
    vector<double> betas2;
    for (int il=0; il<ffnn->getNFedLayers(); ++il){
        // cout << "layer " << il << endl;
        for (int iu1=0; iu1<ffnn->getFedLayer(il)->getNFedUnits(); ++iu1){
            betas.clear();
            for (int ib=BETA_INDEX_OFFSET; ib<ffnn->getFedLayer(il)->getFedUnit(iu1)->getFeeder()->getNBeta(); ++ib) {
                betas.push_back(ffnn->getFedLayer(il)->getFedUnit(iu1)->getFeeder()->getBeta(ib));
}
            for (int iu2=iu1+1; iu2<ffnn->getFedLayer(il)->getNFedUnits(); ++iu2){
                betas2.clear();
                for (int ib=BETA_INDEX_OFFSET; ib<ffnn->getFedLayer(il)->getFedUnit(iu2)->getFeeder()->getNBeta(); ++ib) {
                    betas2.push_back(ffnn->getFedLayer(il)->getFedUnit(iu2)->getFeeder()->getBeta(ib));
}
                const double dot_product = inner_product(begin(betas), end(betas), begin(betas2), 0.0);
                // cout << "    units " << iu1 << ", " << iu2 << "    ->    dot_product = " << dot_product << endl;
                // cout << "        expected " << (((iu1>=ffnn->getLayer(il)->getNUnits()) || (iu2>=ffnn->getLayer(il)->getNUnits())) ? true : false) << endl;
                // cout << endl;
                if ( !( (iu1>=ffnn->getLayer(il)->getNUnits()) || (iu2>=ffnn->getLayer(il)->getNUnits()) ) ){
                    //cout << "_generateSmartBeta (ffnn, u" << iu1 << " u" << iu2 << "): " <<  dot_product << endl;
                    assert( abs(dot_product) < 0.000000001 );
                }
            }
        }
    }

    /*
    // print unit mu sigma
    for (int il=0; il<ffnn->getNLayers(); ++il) {
        cout << "Layer " << il << endl << endl;
        for (int j=0; j<ffnn->getLayer(il)->getNUnits(); ++j) {
            NetworkUnit * u = ffnn->getLayer(il)->getUnit(j);
            cout << "Unit " << j << ", id: " << u->getIdCode() << endl;
            cout << "mu_out " << u->getOutputMu() << ", sigma_out " << u->getOutputSigma() << endl;
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    for (int il=0; il<ffnn->getNFedLayers(); ++il) {
        cout << "Fed Layer " << il << endl << endl;
        for (int j=0; j<ffnn->getFedLayer(il)->getNFedUnits(); ++j) {
            FedUnit * u = ffnn->getFedLayer(il)->getFedUnit(j);
            cout << "Fed Unit " << j << ", id: " << u->getIdCode() << endl;
            cout << "mu_idpv " << u->getIdealProtoMu() << ", sigma_idpv " << u->getIdealProtoSigma() << endl;
            cout << "mu_feed " << u->getFeeder()->getFeedMu() << ", sigma_feed " << u->getFeeder()->getFeedSigma() << endl;
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    delete ffnn;

    return 0;
}

