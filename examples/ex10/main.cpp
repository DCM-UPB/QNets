#include <iostream>
#include <cmath>
#include <random>

#include "NNTrainerGSL.hpp"


/*
  Example: Train FFNN with Feature Map layer

  In this example we want to demonstrate how a feature map layer can be used to
  pre-process the raw input and pass only the relevant information to the Multi-Layer Perceptron.

  Like in the previous example, we consider a gaussian. But this time we use two variables x,y as input,
  as f(x) = exp(-(x-y)²) . To learn this function appropriately, a basic MLP that takes only the raw data requires
  significantly more units and/or layers than before. If you however have the knowledge, that only
  the distance between those two variables is relevant, you can use a minimal feature map layer with
  one distance map unit to reduce the problem for the NN to the same size as before.

  NOTE: In some use cases it is just as easy, but faster, to do the pre-processing on the data yourself (and only once) before
        passing the data to the network. However, when you need the derivatives with respect to the raw input or if you are anyway
        contiously feeding randomly generated data (that you would have to process), the feature-map feature can be very convenient. 
*/

double gaussian(const double x, const double a, const double b) {
    return exp(-a*pow(x-b, 2));
};

// first derivative of gaussian
double gaussian_ddx(const double x, const double a, const double b) {
    return 2.0*a*(b-x) * exp(-a*pow(x-b, 2));
};

// first derivative of gaussian
double gaussian_d2dx(const double x, const double a, const double b) {
    return (pow(2.0*a*(b-x), 2) - 2.0*a) * exp(-a*pow(x-b, 2));
};



int main (void) {
    using namespace std;

    int nhl, nfits = 1;
    double maxchi = 0.0, lambda_r = 0.0, lambda_d1 = 0.0, lambda_d2 = 0.0;
    bool flag_fm = false;
    const bool verbose = false;

    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "How many hidden layers should the FFNN have? (>0) ";
    cin >> nhl;

    int nhu[nhl];
    for (int i=0; i<nhl; ++i) {
        cout << "How many units should hidden layer " << i+1 << " have? (>1) ";
        cin >> nhu[i];
    }
    cout << endl;
    cout << "Do you want to use the pair distance map layer? (0/1)";
    cin >> flag_fm;
    cout << endl;
    cout << endl;

    int nl = nhl + 2;
    if (flag_fm) nl += 1;

    cout << "We generate a FFANN with " << nl << " layers and 3, ";
    if (flag_fm) cout << "2, " << endl;
    for (int i=0; i<nhl; ++i) cout << nhu[i] << ", ";
    cout << "2 units respectively" << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "In the following we use GSL non-linear fit to minimize the mean-squared-distance+regularization of NN vs. target function, i.e. find optimal betas." << endl;
    cout << endl;
    cout << "Please enter the regularization lambda. (e.g. 0.0001) ";
    cin >> lambda_r;
    cout << "Please enter the first derivative lambda. (e.g. 0.1) ";
    cin >> lambda_d1;
    cout << "Please enter the second derivative lambda. (e.g. 0.1) ";
    cin >> lambda_d2;
    cout << "Please enter the the maximum tolerable fit residual. (0 to disable) ";
    cin >> maxchi;
    cout << "Please enter the ";
    if (maxchi > 0) cout << "maximum ";
    cout << "number of fitting runs. (>0) ";
    cin >> nfits;
    cout << endl << endl;
    cout << "Now we find the best fit ... " << endl;
    if (!verbose) cout << "NOTE: You may increase the amount of displayed information by setting verbose to true in the head of main." << endl;
    cout << endl;

    // NON I/O CODE

    // create FFNN
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, nhu[0], 2);
    for (int i = 1; i<nhl; ++i) ffnn->pushHiddenLayer(nhu[i]);

    if (flag_fm) {
        ffnn->pushFeatureMapLayer(2);
        ffnn->getFeatureMapLayer(0)->setNMaps(1,0);
    }

    ffnn->connectFFNN();

    if (flag_fm) ffnn->getFeatureMapLayer(0)->getEDMapUnit(0)->getEDMap()->setParameters(1, 1, 2); // distance of first and second non-offset input

    ffnn->assignVariationalParameters();
    printFFNNStructure(ffnn);

    // create data and config structs
    const int ntraining = 2000; // how many training data points
    const int nvalidation = 1000; // how many validation data points
    const int ntesting = 3000; // how many testing data points
    const int ndata = ntraining + nvalidation + ntesting;
    const int maxn_steps = 100; // maximum number of iterations for least squares solver
    const int maxn_novali = 5; // maximum number of iteration without decreasing validation residual (aka early stopping)
    const int xndim = 2;
    const int yndim = 1;

    NNTrainingData tdata = {ndata, ntraining, nvalidation, xndim, yndim, NULL, NULL, NULL, NULL, NULL}; // we pass NULLs here, since we use tdata.allocate to allocate the data arrays. Alternatively, allocate manually and pass pointers here
    NNTrainingConfig tconfig = {lambda_r, lambda_d1, lambda_d2, maxn_steps, maxn_novali};

    // allocate data arrays
    const bool flag_d1 = lambda_d1>0;
    const bool flag_d2 = lambda_d2>0;
    tdata.allocate(flag_d1, flag_d2);

    // generate the data to be fitted
    const double lb = -10; // lower input boundary for data
    const double ub = 10; // upper input boundary for data
    random_device rdev;
    mt19937_64 rgen = std::mt19937_64(rdev());
    uniform_real_distribution<double> rd(lb,ub);
    for (int i = 0; i < ndata; ++i) {
        tdata.x[i][0] = rd(rgen);
        tdata.x[i][1] = rd(rgen);
        tdata.y[i][0] = gaussian(tdata.x[i][0]-tdata.x[i][1], 1, 0);
        if (flag_d1) {
            tdata.yd1[i][0][0] = gaussian_ddx(tdata.x[i][0]-tdata.x[i][1], 1, 0);
            tdata.yd1[i][0][1] = -gaussian_ddx(tdata.x[i][0]-tdata.x[i][1], 1, 0);
        }
        if (flag_d2) {
            tdata.yd2[i][0][0] = gaussian_d2dx(tdata.x[i][0]-tdata.x[i][1], 1, 0);
            tdata.yd2[i][0][1] = gaussian_d2dx(tdata.x[i][0]-tdata.x[i][1], 1, 0);
        }
        tdata.w[i][0] = 1.0; // our data have no error, so set all weights to 1
        if (verbose) printf ("data: %i %g %g\n", i, tdata.x[i][0]-tdata.x[i][1], tdata.y[i][0]);
    }


    // create trainer and find best fit
    NNTrainerGSL * trainer = new NNTrainerGSL(tdata, tconfig);
    trainer->setNormalization(ffnn); // (optional) setup proper normalization before fitting
    trainer->bestFit(ffnn, nfits, maxchi, verbose ? 2 : 1); // find a fit out of nfits with minimal testing residual

    //

    cout << "Done." << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "Now we print the output/NN to a file. The end." << endl;

    // NON I/O CODE
    trainer->printFitOutput(ffnn, lb, ub, 200, true, true);
    ffnn->storeOnFile("nn.txt");


    // Delete allocations
    delete trainer;
    tdata.deallocate();
    delete ffnn;

    return 0;
    //
}
