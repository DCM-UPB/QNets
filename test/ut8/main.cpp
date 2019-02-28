#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "ffnn/train/NNTrainerGSL.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_vector.h>

#include <cassert>
#include <cstddef> // NULL
#include <iostream>
#include <random>

using namespace std;
using namespace nn_trainer_gsl_details;

void validateJacobian(training_workspace &tws, const double &TINY = 0.00001, const bool &verbose = false)
{
    const int npar = tws.ffnn->getNVariationalParameters(), ndata = tws.ntraining;
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust, *T_fd = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_fdf fdf, fdf_fd;
    gsl_multifit_nlinear_workspace *w, *w_fd;
    const gsl_multifit_nlinear_parameters gsl_params = gsl_multifit_nlinear_default_parameters();

    auto * const fit = new double[npar];
    gsl_vector_view gx = gsl_vector_view_array (fit, npar);

    const bool flag_d = tws.flag_d1 || tws.flag_d2;
    const bool flag_r = tws.flag_r;
    int nresi = 0;

    for (int i=0; i<npar; ++i) { fit[i] = tws.ffnn->getVariationalParameter(i);
}

    // configure fdf object

    if (!flag_r && !flag_d) {
        nresi = calcNData(ndata, tws.yndim);
        fdf.f = ffnn_f_pure;
        fdf.df = ffnn_df_pure;
    }
    else if (flag_r && !flag_d) {
        nresi = calcNData(ndata, tws.yndim, npar);
        fdf.f = ffnn_f_pure_reg;
        fdf.df = ffnn_df_pure_reg;
    }
    else if (!flag_r && flag_d) {
        nresi = calcNData(ndata, tws.yndim, 0, tws.xndim);
        fdf.f = ffnn_f_deriv;
        fdf.df = ffnn_df_deriv;
    }
    else {
        nresi = calcNData(ndata, tws.yndim, npar, tws.xndim);
        fdf.f = ffnn_f_deriv_reg;
        fdf.df = ffnn_df_deriv_reg;
    }

    fdf.fvv = nullptr;
    fdf.n = nresi;
    fdf.p = npar;
    fdf.params = &tws;

    fdf_fd = fdf;
    fdf_fd.df = nullptr; // disable jacobian to use GSL internal finite difference method instead


    // allocate workspace with default parameters, also allocate space for validation
    w = gsl_multifit_nlinear_alloc (T, &gsl_params, nresi, npar);
    w_fd = gsl_multifit_nlinear_alloc (T_fd, &gsl_params, nresi, npar);

    // initialize solver with starting point and calculate jacobians
    gsl_multifit_nlinear_init(&gx.vector, &fdf, w);
    const gsl_matrix * const J = gsl_multifit_nlinear_jac(w);
    const gsl_vector * const f = gsl_multifit_nlinear_residual(w);

    gsl_multifit_nlinear_init(&gx.vector, &fdf_fd, w_fd);
    const gsl_matrix * const J_fd = gsl_multifit_nlinear_jac(w_fd);

    for (int i=0; i<nresi; ++i) {
        if (verbose) { cout << "i=" << i << ": f=" << gsl_vector_get(f, i) << endl;
}
        for (int j=0; j<npar; ++j) {
            if (verbose) { cout << "j=" << j << ": J=" << gsl_matrix_get(J, i, j) << ", J_fd=" << gsl_matrix_get(J_fd, i, j) << " -> diff=" << abs(gsl_matrix_get(J, i, j) - gsl_matrix_get(J_fd, i, j)) << endl;
}
            assert(abs(gsl_matrix_get(J, i, j) - gsl_matrix_get(J_fd, i, j)) < TINY);
        }
    }

    gsl_multifit_nlinear_free(w);
    gsl_multifit_nlinear_free(w_fd);
    delete [] fit;
}

int main(){
    const bool verbose = false;
    const double TINY = 0.00001;

    const int ndim = 2;
    const int xndim = ndim;
    const int nhid = 2;
    const int yndim = ndim;

    // create FFNN
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(xndim+1, nhid, yndim+1);
    ffnn->connectFFNN();
    ffnn->assignVariationalParameters();
    ffnn->addSecondDerivativeSubstrate();

    double fixed_beta[7] = {0.6, -1.1, 0.4, -0.55, 0.45, 1.2, -0.75}; // just some betas
    for (int i=0; i<7; ++i) { ffnn->setBeta(i, fixed_beta[i]);
}

    // allocate data arrays
    const int ntraining = 20;
    const int nvalidation = 20;
    const int ntesting = 20;
    const int ndata = ntraining + nvalidation + ntesting;
    auto ** xdata = new double*[ndata];
    auto ** ydata = new double*[ndata];
    auto *** d1data = new double**[ndata];
    auto *** d2data = new double**[ndata];
    auto ** weights = new double*[ndata];
    for (int i = 0; i<ndata; ++i) {
        xdata[i] = new double[xndim];
        ydata[i] = new double[yndim];
        weights[i] = new double[yndim];
        d1data[i] = new double*[yndim];
        d2data[i] = new double*[yndim];
        for (int j = 0; j<yndim; ++j) {
            d1data[i][j] = new double[xndim];
            d2data[i][j] = new double[xndim];
        }
    }

    // generate the data to be fitted z(x,y) = (-x^3+y^3,x^3-y^3) within [-1,1])
    const double lb = -1;
    const double ub = 1;
    random_device rdev;
    mt19937_64 rgen = std::mt19937_64(rdev());
    uniform_real_distribution<double> rd(lb,ub);
    for (int i=0; i<ndata; ++i) {
        for (int k=0; k<xndim; ++k) {
            xdata[i][k] = rd(rgen);
        }
        for (int j=0; j<yndim; ++j) {
            ydata[i][j] = 0.;
            weights[i][j] = 1.;
            for (int k=0; k<xndim; ++k) {
                ydata[i][j] += pow(xdata[i][k], 3) * ((j==k) ? -1. : 1.);
                d1data[i][j][k] = 3.*pow(xdata[i][k], 2) * ((j==k) ? -1. : 1.);
                d2data[i][j][k] = 6.*xdata[i][k] * ((j==k) ? -1. : 1.);
            }
        }
    };

    // create data/config structs
    const int maxn_steps = 0;
    const int maxn_novali = 0;
    const double lambda_r = 0.75, lambda_d1 = 0.5, lambda_d2 = 0.25;
    NNTrainingData tdata = {ntraining, ntraining, 0, xndim, yndim, xdata, ydata, d1data, d2data, weights};
    NNTrainingConfig tconfig = {0., 0., 0., maxn_steps, maxn_novali}; // initially set all lambdas to 0, i.e. no regularization and derivative residuals
    training_workspace tws;
    tws.copyDatConf(tdata, tconfig);
    tws.ffnn = ffnn;
    tws.ffnn_vderiv = new FeedForwardNeuralNetwork(ffnn);
    tws.ffnn_vderiv->addCrossSecondDerivativeSubstrate();

    // validate Jacobians
    validateJacobian(tws, TINY, verbose); // pure

    tws.lambda_r = lambda_r;
    validateJacobian(tws, TINY, verbose); // pure_reg

    tws.lambda_r = 0.;
    tws.lambda_d1 = lambda_d1;
    validateJacobian(tws, TINY, verbose); // deriv (only d1)

    tws.lambda_d2 = lambda_d2;
    validateJacobian(tws, TINY, verbose); // deriv (with d2)

    tws.lambda_r = lambda_r;
    validateJacobian(tws, TINY, verbose); // deriv_reg


    // delete
    for (int i = 0; i<ndata; ++i) {
        delete [] xdata[i];
        delete [] ydata[i];
        delete [] weights[i];
        for (int j = 0; j<yndim; ++j) {
            delete [] d1data[i][j];
            delete [] d2data[i][j];
        }
        delete [] d1data[i];
        delete [] d2data[i];
    }
    delete [] xdata;
    delete [] ydata;
    delete [] weights;
    delete [] d1data;
    delete [] d2data;

    delete tws.ffnn_vderiv;
    delete ffnn;

    return 0;
};
