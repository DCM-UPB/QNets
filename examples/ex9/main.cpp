#include <cmath>
#include <iostream>
#include <random>

#include "ffnn/train/NNTrainerGSL.hpp"


/*
  Example: Train FFNN via NNTrainerGSL (Levenberg-Marquardt solver)

  In this example we want to demonstrate how a FFNN can be trained via our trainer class to fit target data (in this case a gaussian).

  The fit is achieved by minimizing the mean square error of NN to data (by Levenberg-Marquardt), where the error and gradient are computed from sets of random data points.
  Hence, we have to create the NN and target data (a gaussian), and then let the trainer class do the rest.
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



int main () {
    using namespace std;

    int nhl, nfits = 1;
    double maxchi = 0.0, lambda_r = 0.0, lambda_d1 = 0.0, lambda_d2 = 0.0;
    bool verbose = false;

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

    int nl = nhl + 2;
    cout << "We generate a FFANN with " << nl << " layers and 2, ";
    for (int i=0; i<nhl; ++i) { cout << nhu[i] << ", ";
}
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
    if (maxchi > 0) { cout << "maximum ";
}
    cout << "number of fitting runs. (>0) ";
    cin >> nfits;
    cout << endl << endl;
    cout << "Now we find the best fit ... " << endl;
    if (!verbose) { cout << "NOTE: You may increase the amount of displayed information by setting verbose to true in the head of main." << endl;
}
    cout << endl;

    // NON I/O CODE

    // create FFNN
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, nhu[0], 2);
    for (int i = 1; i<nhl; ++i) { ffnn->pushHiddenLayer(nhu[i]);
}
    ffnn->connectFFNN();
    ffnn->assignVariationalParameters(); // since the trainer works on variational parameter interface, we need to make all betas variational parameters
    //ffnn->assignVariationalParameters(starting_layer_index); // allows you to exclude all betas from layers<starting_layer
    // NOTE: No manual substrate setting has to be done, especially you shouldn't set any variational substrates manually


    // create data and config structs
    const int ntraining = 2000; // how many training data points
    const int nvalidation = 1000; // how many validation data points
    const int ntesting = 3000; // how many testing data points
    const int ndata = ntraining + nvalidation + ntesting;
    const int maxn_steps = 100; // maximum number of iterations for least squares solver
    const int maxn_novali = 5; // maximum number of iteration without decreasing validation residual (aka early stopping)
    const int xndim = 1;
    const int yndim = 1;

    NNTrainingData tdata = {ndata, ntraining, nvalidation, xndim, yndim, nullptr, nullptr, nullptr, nullptr, nullptr}; // we pass NULLs here, since we use tdata.allocate to allocate the data arrays. Alternatively, allocate manually and pass pointers here
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
        tdata.y[i][0] = gaussian(tdata.x[i][0], 1, 0);
        if (flag_d1) { tdata.yd1[i][0][0] = gaussian_ddx(tdata.x[i][0], 1, 0);
}
        if (flag_d2) { tdata.yd2[i][0][0] = gaussian_d2dx(tdata.x[i][0], 1, 0);
}
        tdata.w[i][0] = 1.0; // our data have no error, so set all weights to 1
        if (verbose) { printf ("data: %i %g %g\n", i, tdata.x[i][0], tdata.y[i][0]);
}
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
