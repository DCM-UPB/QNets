#include "NNTrainerGSL.hpp"

#define ACTF_X0 0.0 // target data mean
#define ACTF_XS 1.0 // target data standard deviation
#define ACTF_XD ACTF_XS*3.464101615 //uniform distribution [a,b]: (b-a) = sigma*sqrt(12)
#define ACTF_Y0 0.5 // target output mean
#define ACTF_YD 1.0 // target output interval size


// cost function without regularization and derivative terms
int ffnn_f_pure(const gsl_vector * betas, void * fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingDataGSL *)fit_data)->x;
    const double * const * const y = ((struct NNTrainingDataGSL *)fit_data)->y;
    const double * const * const w = ((struct NNTrainingDataGSL *)fit_data)->w;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta();

    //set new NN betas
    for (int i=0; i<nbeta; ++i){
        ffnn->setBeta(i, gsl_vector_get(betas, i));
    }

    //get difference NN vs data
    for (int i=0; i<n; ++i) {
        ffnn->setInput(x[i]);
        ffnn->FFPropagate();
        for (int j=0; j<ydim; ++j) {
            gsl_vector_set(f, i*ydim + j, w[i][j] * (ffnn->getOutput(j) - y[i][j]));
        }
    }

    return GSL_SUCCESS;
};

// gradient of cost function without regularization and derivative terms
int ffnn_df_pure(const gsl_vector * betas, void * fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingDataGSL *)fit_data)->x;
    const double * const * const w = ((struct NNTrainingDataGSL *)fit_data)->w;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta();

    //set new NN betas
    for (int i=0; i<nbeta; ++i){
        ffnn->setBeta(i, gsl_vector_get(betas, i));
    }

    //calculate cost gradient
    for (int ibeta=0; ibeta<nbeta; ++ibeta) {
        for (int i=0; i<n; ++i) {
            ffnn->setInput(x[i]);
            ffnn->FFPropagate();
            for (int j=0; j<ydim; ++j) {
                gsl_matrix_set(J, i*ydim + j, ibeta, w[i][j] * ffnn->getVariationalFirstDerivative(j, ibeta));
            }
        }
    }

    return GSL_SUCCESS;
};

// cost function with derivative but without regularization
int ffnn_f_deriv(const gsl_vector * betas, void * fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingDataGSL *)fit_data)->x;
    const double * const * const y = ((struct NNTrainingDataGSL *)fit_data)->y;
    const double * const * const * const yd1 = ((struct NNTrainingDataGSL *)fit_data)->yd1;
    const double * const * const * const yd2 = ((struct NNTrainingDataGSL *)fit_data)->yd2;
    const double * const * const w = ((struct NNTrainingDataGSL *)fit_data)->w;
    const double lambda_d1 = ((struct NNTrainingDataGSL *)fit_data)->lambda_d1;
    const double lambda_d2 = ((struct NNTrainingDataGSL *)fit_data)->lambda_d2;
    const bool flag_d1 = ((struct NNTrainingDataGSL *)fit_data)->flag_d1;
    const bool flag_d2 = ((struct NNTrainingDataGSL *)fit_data)->flag_d2;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift = n*ydim, nshift2 = nshift + n*ydim*xdim;
    const double lambda_d1_red = sqrt(lambda_d1), lambda_d2_red = sqrt(lambda_d2);

    //set new NN betas
    for (int i=0; i<nbeta; ++i){
        ffnn->setBeta(i, gsl_vector_get(betas, i));
    }

    //get difference NN vs data
    for (int i=0; i<n; ++i) {
        ffnn->setInput(x[i]);
        ffnn->FFPropagate();
        for (int j=0; j<ydim; ++j) {
            const int ishift = i*ydim;
            const int inshift = ishift + nshift;
            const int inshift2 = ishift + nshift2;

            gsl_vector_set(f, ishift + j, w[i][j] * (ffnn->getOutput(j) - y[i][j]));
            for (int k=0; k<xdim; ++k) {
                if (flag_d1) gsl_vector_set(f, inshift + k*nshift + j, w[i][j] * lambda_d1_red * (ffnn->getFirstDerivative(k, j) - yd1[i][j][k]));
                if (flag_d2) gsl_vector_set(f, inshift2 + k*nshift + j, w[i][j] * lambda_d2_red * (ffnn->getSecondDerivative(k, j) - yd2[i][j][k]));
            }
        }
    }

    return GSL_SUCCESS;
};

// gradient of cost function with derivative but without regularization
int ffnn_df_deriv(const gsl_vector * betas, void * fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingDataGSL *)fit_data)->x;
    const double * const * const w = ((struct NNTrainingDataGSL *)fit_data)->w;
    const double lambda_d1 = ((struct NNTrainingDataGSL *)fit_data)->lambda_d1;
    const double lambda_d2 = ((struct NNTrainingDataGSL *)fit_data)->lambda_d2;
    const bool flag_d1 = ((struct NNTrainingDataGSL *)fit_data)->flag_d1;
    const bool flag_d2 = ((struct NNTrainingDataGSL *)fit_data)->flag_d2;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift = n*ydim, nshift2 = nshift + n*ydim*xdim;
    const double lambda_d1_red = sqrt(lambda_d1), lambda_d2_red = sqrt(lambda_d2);

    //set new NN betas
    for (int i=0; i<nbeta; ++i){
        ffnn->setBeta(i, gsl_vector_get(betas, i));
    }

    //calculate cost gradient
    for (int ibeta=0; ibeta<nbeta; ++ibeta) {
        for (int i=0; i<n; ++i) {
            ffnn->setInput(x[i]);
            ffnn->FFPropagate();
            for (int j=0; j<ydim; ++j) {
                const int ishift = i*ydim;
                const int inshift = ishift + nshift;
                const int inshift2 = ishift + nshift2;

                gsl_matrix_set(J, ishift + j, ibeta, w[i][j] * ffnn->getVariationalFirstDerivative(j, ibeta));
                for (int k=0; k<xdim; ++k) {
                    if(flag_d1) gsl_matrix_set(J, inshift + k*nshift + j, ibeta, w[i][j] * lambda_d1_red * ffnn->getCrossFirstDerivative(k, j, ibeta));
                    if(flag_d2) gsl_matrix_set(J, inshift2 + k*nshift + j, ibeta, w[i][j] * lambda_d2_red * ffnn->getCrossSecondDerivative(k, j, ibeta));
                }
            }
        }
    }

    return GSL_SUCCESS;
};

// cost function for fitting, without derivative but with regularization
int ffnn_f_pure_reg(const gsl_vector * betas, void * fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingDataGSL *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift3 = n*ydim + n*ydim*xdim, n_reg = nshift3 + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_f_pure(betas, fit_data, f);

    //append regularization
    for (int i=nshift3; i<n_reg; ++i) {
        gsl_vector_set(f, i, lambda_r_red * gsl_vector_get(betas, i-nshift3));
    }

    return GSL_SUCCESS;
};

// gradient of cost function without derivatives but with regularization
int ffnn_df_pure_reg(const gsl_vector * betas, void * fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingDataGSL *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift3 = n*ydim + n*ydim*xdim, n_reg = nshift3 + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_df_pure(betas, fit_data, J);

    //append regularization gradient
    for (int i=nshift3; i<n_reg; ++i) {
        for (int j=0; j<nbeta; ++j) {
            gsl_matrix_set(J, i, j, 0.0);
        }
        gsl_matrix_set(J, i, i-nshift3, lambda_r_red);
    }

    return GSL_SUCCESS;
};

// cost function for fitting, with derivative and regularization
int ffnn_f_deriv_reg(const gsl_vector * betas, void * fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingDataGSL *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift3 = n*ydim + n*ydim*xdim, n_reg = nshift3 + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_f_deriv(betas, fit_data, f);

    //append regularization
    for (int i=nshift3; i<n_reg; ++i) {
        gsl_vector_set(f, i, lambda_r_red * gsl_vector_get(betas, i-nshift3));
    }

    return GSL_SUCCESS;
};

// gradient of cost function with derivatives and regularization
int ffnn_df_deriv_reg(const gsl_vector * betas, void * fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingDataGSL *)fit_data)->n;
    const int xdim = ((struct NNTrainingDataGSL *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingDataGSL *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingDataGSL *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * ffnn = ((struct NNTrainingDataGSL *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), nshift3 = n*ydim + n*ydim*xdim, n_reg = nshift3 + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_df_deriv(betas, fit_data, J);

    //append regularization gradient
    for (int i=nshift3; i<n_reg; ++i) {
        for (int j=0; j<nbeta; ++j) {
            gsl_matrix_set(J, i, j, 0.0);
        }
        gsl_matrix_set(J, i, i-nshift3, lambda_r_red);
    }

    return GSL_SUCCESS;
};

// gets called once for every fit iteration
void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w) {
    gsl_vector *f = gsl_multifit_nlinear_residual(w);
    gsl_vector *x = gsl_multifit_nlinear_position(w);
    double rcond = 0.0;

    // compute reciprocal condition number of J(x)
    gsl_multifit_nlinear_rcond(&rcond, w);

    fprintf(stderr, "iter %zu: cond(J) = %8.4f, |f(x)| = %.4f\n", iter, 1.0 / rcond, gsl_blas_dnrm2(f));

    for (size_t i=0; i<x->size; ++i) fprintf(stderr, "b%zu: %f, ", i,  gsl_vector_get(x, i));
    fprintf(stderr, "\n");
};


/* backup of old initialization/normalization code

        _ndata = ndata;

        _flag_d1 = flag_d1;
        _flag_d2 = flag_d2;

        _ffnn = new FeedForwardNeuralNetwork(2, nhunits[0], 2);
        for (int i = 1; i<nhlayer; ++i) _ffnn->pushHiddenLayer(nhunits[i]);
        _ffnn->connectFFNN();
        if (flag_d1) _ffnn->addFirstDerivativeSubstrate();
        if (flag_d2) _ffnn->addSecondDerivativeSubstrate();
        _ffnn->addVariationalFirstDerivativeSubstrate();
        _ffnn->addCrossFirstDerivativeSubstrate();
        _ffnn->addCrossSecondDerivativeSubstrate();

        _npar = _ffnn->getNBeta();

        _xdata = new double[ndata]; // we need own copies to normalize the data
        _ydata = new double[ndata];
        _d1data = new double[ndata];
        _d2data = new double[ndata];
        _weights = weights; // allow to account for noisy data by weighing in the error

        _lambda_d1 = lambda_d1;
        _lambda_d2 = lambda_d2;
        _lambda_r = lambda_r;

        // calculate values for normalization
        auto mimax = minmax_element(xdata, xdata + ndata);
        auto mimay = minmax_element(ydata, ydata + ndata);
        double x0 = 0.5*(*mimax.first + *mimax.second);
        double y0 = 0.5*(*mimay.first + *mimay.second);
        double xd = *mimax.second - *mimax.first;
        double yd = *mimay.second - *mimay.first;

        _xscale = ACTF_XD/xd;
        _yscale = ACTF_YD/yd;
        _xshift = ACTF_X0 - x0;
        _yshift = ACTF_Y0 - y0;

        for (int i = 0; i < ndata; ++i) {
            _xdata[i] = (xdata[i] + _xshift) * _xscale;
            _ydata[i] = (ydata[i] + _yshift) * _yscale;
            _d1data[i] = d1data[i] * _yscale / _xscale;
            _d2data[i] = d2data[i] * _yscale / pow(_xscale, 2);
        }
    }
*/

void NNTrainerGSL::findFit(const int nsteps, double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const bool verbose) {

    //   Fit NN to data with following parameters:
    //   nsteps : number of fitting iterations
    //   nfits : maximum number of fits to achieve good fit
    //   maxchi : maximum tolerable residual to consider a fit good
    //   verbose: print verbose output while fitting


    int npar = _ffnn->getNBeta(), ndata = _tdata->n;
    const gsl_multifit_nlinear_type *T_full = gsl_multifit_nlinear_trust, *T_noreg = gsl_multifit_nlinear_trust, *T_pure = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_fdf fdf_full, fdf_noreg, fdf_pure;
    gsl_multifit_nlinear_workspace * w_full, * w_noreg, * w_pure;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

    gsl_vector *f;
    gsl_matrix *J;
    gsl_matrix * covar = gsl_matrix_alloc (npar, npar);

    double fit_init[npar];
    gsl_vector_view gx = gsl_vector_view_array (fit_init, npar);
    double chisq, chisq0, chi0;
    double chisq_noreg, chisq_pure;
    double c;
    const int dof = ndata - npar;
    int ndata_noreg, ndata_full;
    int status, info;

    const double xtol = 0.0;
    const double gtol = 0.0;
    const double ftol = 0.0;

    const bool flag_d = _tdata->flag_d1 || _tdata->flag_d2;

    // first the pure fdf
    fdf_pure.f = ffnn_f_pure;
    fdf_pure.df = ffnn_df_pure;
    fdf_pure.fvv = NULL;
    fdf_pure.n = ndata;
    fdf_pure.p = npar;
    fdf_pure.params = &_tdata;

    if (flag_d) {
        ndata_noreg = ndata * _tdata->ydim + ndata * _tdata->ydim * _tdata->xdim;

        // deriv fdf without regularization
        fdf_noreg.f = ffnn_f_deriv;
        fdf_noreg.df = ffnn_df_deriv;
        fdf_noreg.fvv = NULL;
        fdf_noreg.n = ndata_noreg;
        fdf_noreg.p = npar;
        fdf_noreg.params = &_tdata;
    }
    else {
        ndata_noreg = ndata;
        fdf_noreg = fdf_pure;
    };

    if (_tdata->flag_r) {
        ndata_full = ndata_noreg + npar;

        if (flag_d) {
            // deriv with regularization
            fdf_full.f = ffnn_f_deriv_reg;
            fdf_full.df = ffnn_df_deriv_reg;
            fdf_full.fvv = NULL;
            fdf_full.n = ndata_full;
            fdf_full.p = npar;
            fdf_full.params = &_tdata;
        }
        else {
            // pure fdf with regularization
            fdf_full.f = ffnn_f_pure_reg;
            fdf_full.df = ffnn_df_pure_reg;
            fdf_full.fvv = NULL;
            fdf_full.n = ndata_full;
            fdf_full.p = npar;
            fdf_full.params = &_tdata;
        }
    }
    else {
        ndata_full = ndata_noreg;
        fdf_full = fdf_noreg;
    };

    // allocate workspace with default parameters
    w_full = gsl_multifit_nlinear_alloc (T_full, &fdf_params, ndata_full, npar);
    w_noreg = gsl_multifit_nlinear_alloc (T_noreg, &fdf_params, ndata_noreg, npar);
    w_pure = gsl_multifit_nlinear_alloc (T_pure, &fdf_params, ndata, npar);

    // initialize solver with starting point
    gsl_multifit_nlinear_init(&gx.vector, &fdf_full, w_full);

    // compute initial cost function
    f = gsl_multifit_nlinear_residual(w_full);
    gsl_blas_ddot(f, f, &chisq0);
    chi0 = sqrt(chisq0);

    // solve the system with a maximum of nsteps iterations
    if (verbose) status = gsl_multifit_nlinear_driver(nsteps, xtol, gtol, ftol, callback, NULL, &info, w_full);
    else status = gsl_multifit_nlinear_driver(nsteps, xtol, gtol, ftol, NULL, NULL, &info, w_full);

    // compute covariance of best fit parameters
    J = gsl_multifit_nlinear_jac(w_full);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    // compute final cost
    gsl_blas_ddot(f, f, &chisq);
    resi_full = sqrt(chisq);
    c = GSL_MAX_DBL(1, sqrt(chisq / dof));

    // unregularized cost calculation
    for (int i = 0; i<npar; ++i) {
        fit[i] = gsl_vector_get(w_full->x, i);
        err[i] = c*sqrt(gsl_matrix_get(covar,i,i));

        fit_init[i] = fit[i]; // to calculate the other residuals
    }
    gsl_multifit_nlinear_init(&gx.vector, &fdf_noreg, w_noreg);
    f = gsl_multifit_nlinear_residual(w_noreg);
    gsl_blas_ddot(f, f, &chisq_noreg);
    resi_noreg = sqrt(chisq_noreg);

    // pure (no deriv, no reg) cost calculation
    gsl_multifit_nlinear_init(&gx.vector, &fdf_pure, w_pure);
    f = gsl_multifit_nlinear_residual(w_pure);
    gsl_blas_ddot(f, f, &chisq_pure);
    resi_pure = sqrt(chisq_pure);

    if (verbose) {
        fprintf(stderr, "summary from method '%s/%s'\n", gsl_multifit_nlinear_name(w_full), gsl_multifit_nlinear_trs_name(w_full));
        fprintf(stderr, "number of iterations: %zu\n", gsl_multifit_nlinear_niter(w_full));
        fprintf(stderr, "function evaluations: %zu\n", fdf_full.nevalf);
        fprintf(stderr, "Jacobian evaluations: %zu\n", fdf_full.nevaldf);
        fprintf(stderr, "reason for stopping: %s\n", (info == 1) ? "small step size" : "small gradient");
        fprintf(stderr, "initial |f(x)| = %f\n", chi0);
        fprintf(stderr, "final   |f(x)| = %f\n", resi_full);
        fprintf(stderr, "w/o reg |f(x)| = %f\n", resi_noreg);
        fprintf(stderr, "pure    |f(x)| = %f\n", resi_pure);
        fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

        for(int i=0; i<npar; ++i) fprintf(stderr, "b%i      = %.5f +/- %.5f\n", i, fit[i], err[i]);

        fprintf(stderr, "status = %s\n", gsl_strerror (status));
    }

    gsl_multifit_nlinear_free(w_full);
    gsl_multifit_nlinear_free(w_noreg);
    gsl_multifit_nlinear_free(w_pure);
    gsl_matrix_free(covar);



}


/*

    // compute fit distance for best betas
    double getFitDistance() {
        double dist = 0.0;
        for(int i=0; i<_ndata; ++i) {
            _ffnn->setInput(0, _xdata[i]);
            _ffnn->FFPropagate();
            dist += pow(_ydata[i]-_ffnn->getOutput(0), 2);
        }
        return dist / _ndata / pow(_yscale, 2);
    }

    // compare NN to data from index i0 to ie in increments di
    void compareFit(const int i0=0, const int ie=-1, const int di = 1) {
        using namespace std;

        const int realie = (ie<0)? _ndata-1:ie; //set default ie although _ndata is not const

        int j=i0;
        while(j<_ndata && j<=realie){
            _ffnn->setInput(0, _xdata[j]);
            _ffnn->FFPropagate();
            cout << "x: " << _xdata[j] / _xscale - _xshift << " f(x): " << _ydata[j] / _yscale - _yshift << " nn(x): " << _ffnn->getOutput(0) / _yscale - _yshift << endl;
            j+=di;
        }
        cout << endl;
    }

    // print output of fitted NN to file
    void printFitOutput(const double &min, const double &max, const int &npoints, const bool &print_d1 = false, const bool &print_d2 = false) {
        double base_input = 0.0;
        writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getOutput", "v.txt", _xscale, _yscale, _xshift, _yshift);
        if (print_d1 && _flag_d1) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getFirstDerivative", "d1.txt", _xscale, _yscale, _xshift, _yshift);
        if (print_d2 && _flag_d2) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getSecondDerivative", "d2.txt", _xscale, _yscale, _xshift, _yshift);
    }

    // store fitted NN in file
    void printFitNN() {_ffnn->storeOnFile("nn.txt");}

    FeedForwardNeuralNetwork * getFFNN() {return _ffnn;}
};
*/
/*
int main (void) {
    using namespace std;

    NNFitter1D * fitter;

    double lb = -10;
    double ub = 10;
    int ndata = 2001;
    double xdata[ndata];
    double ydata[ndata];
    double d1data[ndata];
    double d2data[ndata];
    double weights[ndata];

    int nl, nhl, nhu[2], nfits = 1;
    double maxchi = 0.0, lambda_r = 0.0, lambda_d1 = 0.0, lambda_d2 = 0.0;

    bool verbose = false;

    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "How many units should the first hidden layer(s) have? ";
    cin >> nhu[0];
    cout << "How many units should the second hidden layer(s) have? (<=1 for none) ";
    cin >> nhu[1];
    cout << endl;

    // NON I/O CODE
    nl = (nhu[1]>1)? 4:3;
    nhl = nl-2;
    //

    cout << "We generate a FFANN with " << nl << " layers and 2, " << nhu[0];
    if (nhu[1]>0) { cout << ", " << nhu[1];}
    cout << ", 2 units respectively" << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "In the following we use GSL non-linear fit to minimize the mean-squared-distance+regularization of NN vs. target function, i.e. find optimal betas." << endl;
    cout << endl;
    cout << "Please enter the first derivative lambda. ";
    cin >> lambda_d1;
    cout << "Please enter the second derivative lambda. ";
    cin >> lambda_d2;
    cout << "Please enter the regularization lambda. ";
    cin >> lambda_r;
    cout << "Please enter the the maximum tolerable fit residual. ";
    cin >> maxchi;
    cout << "Please enter the maximum number of fitting runs. ";
    cin >> nfits;
    cout << endl << endl;
    cout << "Now we find the best fit ... " << endl;


    // NON I/O CODE

    // this is the data to be fitted
    double dx = (ub-lb) / (ndata-1);
    for (int i = 0; i < ndata; ++i) {
        xdata[i] = lb + i*dx;
        ydata[i] = gaussian(xdata[i], 1, 0);
        d1data[i] = gaussian_ddx(xdata[i], 1, 0);
        d2data[i] = gaussian_d2dx(xdata[i], 1, 0);
        weights[i] = 1.0; // our data have no error, so set all weights to 1
        if (verbose) printf ("data: %i %g %g\n", i, xdata[i], ydata[i]);
    };


    fitter = new NNFitter1D(nhl, nhu, ndata, xdata, ydata, d1data, d2data, weights, lambda_d1, lambda_d2, lambda_r, true, true);
    fitter->findFit(100, nfits, maxchi, false);
    //

    cout << "Done." << endl;
    cout << "========================================================================" << endl;
    cout << endl;
    cout << "Finally we compare the best fit NN to the target function:" << endl << endl;

    // NON I/O CODE
    fitter->compareFit(0, ndata, 100);
    //

    cout << endl;
    cout << "And print the output/NN to a file. The end." << endl;

    // NON I/O CODE
    fitter->printFitOutput(-10, 10, 200, true, true);
    fitter->printFitNN();
    //

    delete fitter;

    return 0;
}
*/
