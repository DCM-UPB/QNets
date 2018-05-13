#include "NNTrainerGSL.hpp"

// cost function without regularization and derivative terms
int ffnn_f_pure(const gsl_vector * betas, void * const fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingData *)fit_data)->x;
    const double * const * const y = ((struct NNTrainingData *)fit_data)->y;
    const double * const * const w = ((struct NNTrainingData *)fit_data)->w;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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
int ffnn_df_pure(const gsl_vector * betas, void * const fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingData *)fit_data)->x;
    const double * const * const w = ((struct NNTrainingData *)fit_data)->w;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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
int ffnn_f_deriv(const gsl_vector * betas, void * const fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int xdim = ((struct NNTrainingData *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingData *)fit_data)->x;
    const double * const * const y = ((struct NNTrainingData *)fit_data)->y;
    const double * const * const * const yd1 = ((struct NNTrainingData *)fit_data)->yd1;
    const double * const * const * const yd2 = ((struct NNTrainingData *)fit_data)->yd2;
    const double * const * const w = ((struct NNTrainingData *)fit_data)->w;
    const double lambda_d1 = ((struct NNTrainingData *)fit_data)->lambda_d1;
    const double lambda_d2 = ((struct NNTrainingData *)fit_data)->lambda_d2;
    const bool flag_d1 = ((struct NNTrainingData *)fit_data)->flag_d1;
    const bool flag_d2 = ((struct NNTrainingData *)fit_data)->flag_d2;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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
                gsl_vector_set(f, inshift + k*nshift + j, flag_d1? w[i][j] * lambda_d1_red * (ffnn->getFirstDerivative(j, k) - yd1[i][j][k]) : 0.0);
                gsl_vector_set(f, inshift2 + k*nshift + j, flag_d2? w[i][j] * lambda_d2_red * (ffnn->getSecondDerivative(j, k) - yd2[i][j][k]) : 0.0);
            }
        }
    }

    return GSL_SUCCESS;
};

// gradient of cost function with derivative but without regularization
int ffnn_df_deriv(const gsl_vector * betas, void * const fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int xdim = ((struct NNTrainingData *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double * const * const x = ((struct NNTrainingData *)fit_data)->x;
    const double * const * const w = ((struct NNTrainingData *)fit_data)->w;
    const double lambda_d1 = ((struct NNTrainingData *)fit_data)->lambda_d1;
    const double lambda_d2 = ((struct NNTrainingData *)fit_data)->lambda_d2;
    const bool flag_d1 = ((struct NNTrainingData *)fit_data)->flag_d1;
    const bool flag_d2 = ((struct NNTrainingData *)fit_data)->flag_d2;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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
                    gsl_matrix_set(J, inshift + k*nshift + j, ibeta, flag_d1? w[i][j] * lambda_d1_red * ffnn->getCrossFirstDerivative(j, k, ibeta) : 0.0);
                    gsl_matrix_set(J, inshift2 + k*nshift + j, ibeta, flag_d2? w[i][j] * lambda_d2_red * ffnn->getCrossSecondDerivative(j, k, ibeta) : 0.0);
                }
            }
        }
    }

    return GSL_SUCCESS;
};

// cost function for fitting, without derivative but with regularization
int ffnn_f_pure_reg(const gsl_vector * betas, void * const fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const double lambda_r = ((struct NNTrainingData *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), n_reg = n + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_f_pure(betas, fit_data, f);

    //append regularization
    for (int i=n; i<n_reg; ++i) {
        gsl_vector_set(f, i, lambda_r_red * gsl_vector_get(betas, i-n));
    }

    return GSL_SUCCESS;
};

// gradient of cost function without derivatives but with regularization
int ffnn_df_pure_reg(const gsl_vector * betas, void * const fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const double lambda_r = ((struct NNTrainingData *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

    const int nbeta = ffnn->getNBeta(), n_reg = n + nbeta;
    const double lambda_r_red = sqrt(lambda_r / nbeta);

    ffnn_df_pure(betas, fit_data, J);

    //append regularization gradient
    for (int i=n; i<n_reg; ++i) {
        for (int j=0; j<nbeta; ++j) {
            gsl_matrix_set(J, i, j, 0.0);
        }
        gsl_matrix_set(J, i, i-n, lambda_r_red);
    }

    return GSL_SUCCESS;
};

// cost function for fitting, with derivative and regularization
int ffnn_f_deriv_reg(const gsl_vector * betas, void * const fit_data, gsl_vector * f) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int xdim = ((struct NNTrainingData *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingData *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * const ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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
int ffnn_df_deriv_reg(const gsl_vector * betas, void * const fit_data, gsl_matrix * J) {
    const int n = ((struct NNTrainingData *)fit_data)->n;
    const int xdim = ((struct NNTrainingData *)fit_data)->xdim;
    const int ydim = ((struct NNTrainingData *)fit_data)->ydim;
    const double lambda_r = ((struct NNTrainingData *)fit_data)->lambda_r;
    FeedForwardNeuralNetwork * ffnn = ((struct NNTrainingData *)fit_data)->ffnn;

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


void NNTrainerGSL::findFit(double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const int nsteps, const int verbose) {

    //   Fit NN with the following passed variables:
    //   fit: holds the to be fitted variables, i.e. betas
    //   err: holds the corresponding fit error
    //   resi_full: holds the full residual value, including all terms
    //   resi_noreg: holds the residual value without regularization
    //   resi_pure: holds the function-only residual value
    //
    //   and with the following parameters:
    //   nsteps : number of fitting iterations
    //   verbose: print verbose output while fitting




    int npar = _ffnn->getNBeta(), ndata = _tdata->n;
    const gsl_multifit_nlinear_type *T_full = gsl_multifit_nlinear_trust, *T_noreg = gsl_multifit_nlinear_trust, *T_pure = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_fdf fdf_full, fdf_noreg, fdf_pure;
    gsl_multifit_nlinear_workspace * w_full, * w_noreg, * w_pure;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

    gsl_vector *f;
    gsl_matrix *J;
    gsl_matrix * covar = gsl_matrix_alloc (npar, npar);

    gsl_vector_view gx = gsl_vector_view_array (fit, npar);
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
    fdf_pure.params = _tdata;

    if (flag_d) {
        ndata_noreg = ndata * _tdata->ydim + 2*(ndata * _tdata->ydim * _tdata->xdim);

        // deriv fdf without regularization
        fdf_noreg.f = ffnn_f_deriv;
        fdf_noreg.df = ffnn_df_deriv;
        fdf_noreg.fvv = NULL;
        fdf_noreg.n = ndata_noreg;
        fdf_noreg.p = npar;
        fdf_noreg.params = _tdata;
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
            fdf_full.params = _tdata;
        }
        else {
            // pure fdf with regularization
            fdf_full.f = ffnn_f_pure_reg;
            fdf_full.df = ffnn_df_pure_reg;
            fdf_full.fvv = NULL;
            fdf_full.n = ndata_full;
            fdf_full.p = npar;
            fdf_full.params = _tdata;
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
    if (verbose > 1) status = gsl_multifit_nlinear_driver(nsteps, xtol, gtol, ftol, callback, NULL, &info, w_full);
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

    if (verbose > 1) {
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



};


/*


};
*/
